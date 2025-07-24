import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


ENV_NAME = "CartPole-v1"
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
LR = 3e-4
VF_COEF = 0.5
ENT_COEF = 0.01
BATCH_SIZE = 2048
MINI_BATCH_SIZE = 256
UPDATE_EPOCHS = 10
MAX_TRAIN_STEPS = 200_000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64,64), nn.Tanh()
        )
        self.pi = nn.Linear(64, action_dim)
        self.v = nn.Linear(64,1)
    
    def forward(self, x):
        x = self.shared(x)
        return self.pi(x), self.v(x)
    
    def act(self, s):
        logits, _ = self.forward(s)
        dist = Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a), dist.entropy()

    def evaluate(self, s, a):
        logits, v = self.forward(s)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(a)
        entropy = dist.entropy()
        return logp, entropy, v.squeeze(-1)    
    

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.logprobs = []
        self.values = []
    
    def add(self, s, a , r, d, logp, v):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.dones.append(d)
        self.logprobs.append(logp)
        self.values.append(v)
    
    def clear(self):
        self.__init__()

def gae_advantages(rewards, values, dones, gamma, lam):
    adv = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for i in reversed(range(len(rewards))):
        mask = 1.0 - dones[i]
        # TD误差
        delta = rewards[i] + gamma * values[i+1] * mask - values[i]
        gae = delta + gamma * lam * mask * gae
        adv[i] = gae
    returns = adv + values[:-1]
    return adv, returns

def shape_reward(orginal_reward, next_state, env):
    x, x_dot, theta, theta_dot = next_state
    x_thr = env.unwrapped.x_threshold
    th_thr = env.unwrapped.theta_threshold_radians
    x_norm = abs(x) / x_thr
    theta_norm = abs(theta) / th_thr
    shaping = 0.2 * (1.0 - x_norm) + 0.8 * (1.0 - theta_norm)
    return orginal_reward + 0.5 * shaping
    
def train(save_path="ppo_cartpole_v1.pth"):
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    ac = ActorCritic(state_dim, action_dim).to(DEVICE)
    opt = optim.Adam(ac.parameters(), lr=LR)
    buffer = RolloutBuffer()

    state, _ = env.reset()
    total_steps = 0
    episode = 0
    while total_steps < MAX_TRAIN_STEPS:
        
        buffer.clear()
        for _ in range(BATCH_SIZE):

            state = np.array(state)
            s_tensor = torch.from_numpy(state).float().to(DEVICE)
            with torch.no_grad():
                a, logp, _ = ac.act(s_tensor)
                v = ac.forward(s_tensor)[1].item()
            act = a.item()
            next_state, reward, terminated, truncated, _ = env.step(act)
            done =terminated or truncated

            reward_mod = shape_reward(reward, next_state, env)

            buffer.add(state, act, reward_mod, float(done), logp.item(), v)

            state = next_state
            total_steps += 1
            if done:
                episode += 1
                state, _ = env.reset()
        
        with torch.no_grad():
            state = np.array(state)
            s_tensor = torch.from_numpy(state).float().to(DEVICE)
            last_v = ac.forward(s_tensor)[1].item()

        
        rewards = np.array(buffer.rewards, dtype=np.float32)
        dones   = np.array(buffer.dones, dtype=np.float32)
        values  = np.array(buffer.values + [last_v], dtype=np.float32)
        adv, rets = gae_advantages(rewards, values, dones, GAMMA, LAMBDA)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        
        states = torch.tensor(np.array(buffer.states), dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(buffer.actions, dtype=torch.int64, device=DEVICE)
        old_logp = torch.tensor(buffer.logprobs, dtype=torch.float32, device=DEVICE)
        advantages = torch.tensor(adv, dtype=torch.float32, device=DEVICE)
        returns = torch.tensor(rets, dtype=torch.float32, device=DEVICE)

        
        idx = np.arange(BATCH_SIZE)
        for _ in range(UPDATE_EPOCHS):
            np.random.shuffle(idx)
            for start in range(0,BATCH_SIZE,MINI_BATCH_SIZE):
                end = start + MINI_BATCH_SIZE
                mb_idx = idx[start:end]

                logp, entropy, value = ac.evaluate(states[mb_idx], actions[mb_idx])
                ratio = torch.exp(logp - old_logp[mb_idx])

                surr1 = ratio * advantages[mb_idx]
                surr2 = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * advantages[mb_idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = (returns[mb_idx] - value).pow(2).mean()
                entropy_loss = entropy.mean()

                loss = policy_loss + VF_COEF*value_loss - ENT_COEF*entropy_loss

                opt.zero_grad()
                loss.backward()
                opt.step()

        if episode % 10 == 0:
            print(f"Steps:{total_steps} Ep:{episode} Loss:{loss.item():.3f} "
                  f"V:{value_loss.item():.3f} P:{policy_loss.item():.3f}")
            
    torch.save(ac.state_dict(), save_path)
    print(f"模型已保存到{save_path}")
    env.close()


def test(model_path="ppo_cartpole.pth", episodes=10, render=False):
    env = gym.make(ENV_NAME, render_mode="human" if render else None)
    ac  = ActorCritic(env.observation_space.shape[0],
                      env.action_space.n).to(DEVICE)

    ac.load_state_dict(torch.load(model_path, map_location=DEVICE))
    ac.eval()

    for ep in range(1, episodes+1):
        state, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:

            s_tensor = torch.from_numpy(state).float().to(DEVICE)
            with torch.no_grad():
                logits, _ = ac.forward(s_tensor)
                dist = Categorical(logits=logits)
                action = dist.sample().item()

            # 交互
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            if render:
                env.render()

        print(f"Test Episode {ep}: Reward = {total_reward:.1f}")

    env.close()


if __name__ == "__main__":

    train("ppo_cartpole.pth")

    test("ppo_cartpole.pth", episodes=5, render=True)
