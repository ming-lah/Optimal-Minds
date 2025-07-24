import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

import torch
print(torch.cuda.is_available()) 


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)  # 输出层

    def forward(self, state):
        x = torch.relu(self.fc1(state))  # ReLU激活
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # 返Q值


class DQNAgent:
    def __init__(self, env, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, alpha=0.001, batch_size=64, memory_size=10000):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)  # 经验回放池

        device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        self.device = device

        self.q_network = QNetwork(env.observation_space.shape[0], env.action_space.n).to(self.device)  # 当前Q网络
        self.target_network = QNetwork(env.observation_space.shape[0], env.action_space.n).to(self.device)  # 目标Q网络
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=alpha)  # 优化器
        self.update_target_network()  # 初始化目标网络

    def choose_action(self, state):
        state = np.array(state)
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        q_values = self.q_network(state)
        return torch.argmax(q_values).item()

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def sample_batch(self):
        return random.sample(self.memory, self.batch_size)
    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = self.sample_batch()
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states)
        next_states = np.array(next_states)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # 计算当前Q网络输出的Q值
        q_values = self.q_network(states).gather(1, actions.view(-1, 1)).squeeze(1)

        # 计算目标q值
        next_q_values = self.target_network(next_states).max(1)[0]
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

# 保存模型
def save_model(agent, filename="dqn_cartpole.pth"):
    torch.save({
        'q_network': agent.q_network.state_dict(),
        'target_network': agent.target_network.state_dict(),
        'epsilon': agent.epsilon,
    }, filename)
    print(f"Model saved to {filename}")


# DQN训练过程
def train_dqn(episodes=1000):
    env = gym.make('CartPole-v1')
    agent = DQNAgent(env)

    for episode in range(episodes):
        state, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store_experience(state, action, reward, next_state, done)
            agent.train()  # 更新Q网络
            state = next_state
            total_reward += reward

        if episode % 10 == 0:
            agent.update_target_network()
            print(f"Episode {episode}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")
    
    save_model(agent)

train_dqn(1000)


def load_model(agent, filename="dqn_cartpole.pth"):
    checkpoint = torch.load(filename)
    agent.q_network.load_state_dict(checkpoint['q_network'])
    agent.target_network.load_state_dict(checkpoint['target_network'])
    agent.epsilon = checkpoint['epsilon']
    print(f"Model loaded from {filename}")

def run_inference(agent, env):
    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        state = next_state
        env.render()

    print(f"Total reward in the test run: {total_reward}")
    env.close()


def test_dqn():
    env = gym.make('CartPole-v1')
    agent = DQNAgent(env)
    load_model(agent)
    run_inference(agent, env)

test_dqn()


