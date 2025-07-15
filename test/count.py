import json
import numpy as np
import matplotlib.pyplot as plt

# Load the uploaded map file
file_path = 'map.json'  # Adjust the path as needed
with open(file_path, 'r', encoding='utf-8') as f:
    fmap = json.load(f)

GRID_SIZE = 64  # 64x64 grid
N_STATES = GRID_SIZE * GRID_SIZE

# Identify endpoint states: next_state with done==True
endpoints = set()
for s_str, actions in fmap.items():
    for a_str, trans in actions.items():
        next_state, reward, done = trans
        if done:
            endpoints.add(next_state)

# If no explicit endpoints found, fallback to reward>0 but done==False
if not endpoints:
    for s_str, actions in fmap.items():
        for a_str, trans in actions.items():
            next_state, reward, done = trans
            if reward > 0:
                endpoints.add(next_state)

# BFS to compute shortest distance from endpoints
INF = 1e9
dist = np.full(N_STATES, INF, dtype=int)
from collections import deque
queue = deque()

for ep in endpoints:
    dist[ep] = 0
    queue.append(ep)

# Build adjacency list: from state, which states can reach it
reverse_adj = [[] for _ in range(N_STATES)]
for s_str, actions in fmap.items():
    s = int(s_str)
    for a_str, trans in actions.items():
        s_next, reward, done = trans
        reverse_adj[s_next].append(s)

while queue:
    u = queue.popleft()
    for prev in reverse_adj[u]:
        if dist[prev] == INF:
            dist[prev] = dist[u] + 1
            queue.append(prev)

# Reshape and transpose to swap X/Y
dist_matrix = dist.reshape((GRID_SIZE, GRID_SIZE)).T  # Swap X/Y

# Mask unreachable states for better visualization
masked = np.ma.masked_where(dist_matrix >= INF, dist_matrix)

plt.figure(figsize=(8, 8))
img = plt.imshow(masked, origin='lower')
plt.colorbar(img, label='Shortest steps to endpoint')

# Plot endpoints with swapped X/Y
ep_coords = [(ep // GRID_SIZE, ep % GRID_SIZE) for ep in endpoints]

if ep_coords:
    xs, ys = zip(*ep_coords)
    plt.scatter(xs, ys, marker='o', s=50, label='Endpoint(s)', color='red')

plt.title('Map Distance Heatmap (to endpoint)')
plt.legend(loc='upper right')
plt.xlabel('X (column)')
plt.ylabel('Y (row)')
plt.tight_layout()

plt.savefig('heatmap.png', dpi=300)
plt.show()










print("======================new test======================")

import numpy as np

# data of parameters: state_size4096, episodes100, gamma0.9, theta0.001

gammma = 0.9
theta = 0.001
episodes = 100
action_size = 4
state_size = 4096


class Algorithm:
    def __init__(self, gamma, theta, episodes, state_size, action_size, logger):
        self.state_size = state_size
        self.gamma = gamma
        self.theta = theta
        self.episodes = episodes
        self.action_size = action_size
        self.logger = logger
    
    def value_iteration(self, F):
        V = np.zeros(self.state_size)
        i = 0
        while i < self.episodes:
            delta = 0
            
            for state in range(self.state_size):
                v = V[state]

                V[state] = max(self._get_value(state, action, F, V) for action in range(self.action_size))

                delta = max(delta, abs(v - V[state]))

            if delta < self.theta:
                self.episodes_self = i
                break

            policy = self.policy_improvement(self.q_value_iteration(V, F))
            if i% 10 == 0:
                self.logger.info("Iteration {}".format(i))
            i += 1
        
        return V, policy
    