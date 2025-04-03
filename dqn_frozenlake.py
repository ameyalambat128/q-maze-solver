import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import time

# --- Hyperparameters ---
ENV_NAME = "FrozenLake-v1"
NUM_EPISODES = 10000
MAX_STEPS_PER_EPISODE = 100
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 10000
TARGET_UPDATE_FREQ = 100  # Update target network every N episodes

# --- Environment Setup ---
env = gym.make(ENV_NAME, is_slippery=False, render_mode=None)
num_states = env.observation_space.n
num_actions = env.action_space.n

# One-hot encoding for discrete states


def one_hot_state(state, num_states):
    vec = np.zeros(num_states, dtype=np.float32)
    vec[state] = 1.0
    return vec

# --- Define the Q-Network ---


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
online_net = DQN(num_states, num_actions).to(device)
target_net = DQN(num_states, num_actions).to(device)
target_net.load_state_dict(online_net.state_dict())
target_net.eval()

optimizer = optim.Adam(online_net.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# --- Replay Buffer ---


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

# --- Training Loop ---
epsilon = EPSILON_START
rewards_per_episode = []

print("Starting DQN Training...")
start_time = time.time()

for episode in range(NUM_EPISODES):
    state, info = env.reset()
    state = int(state)
    state_vec = one_hot_state(state, num_states)
    total_reward = 0
    done = False

    for step in range(MAX_STEPS_PER_EPISODE):
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = online_net(state_tensor)
            action = torch.argmax(q_values).item()

        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = int(next_state)
        done = terminated or truncated

        # Optional: slight penalty for falling in a hole
        if terminated and reward == 0.0:
            reward = -0.1

        next_state_vec = one_hot_state(next_state, num_states)
        replay_buffer.push(state_vec, action, reward, next_state_vec, done)

        state_vec = next_state_vec
        total_reward += reward
        if done:
            break

    rewards_per_episode.append(total_reward)
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    # Update online network using a mini-batch from replay buffer
    if len(replay_buffer) >= BATCH_SIZE:
        batch = replay_buffer.sample(BATCH_SIZE)
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = zip(
            *batch)

        batch_states = torch.FloatTensor(batch_states).to(device)
        batch_actions = torch.LongTensor(batch_actions).unsqueeze(1).to(device)
        batch_rewards = torch.FloatTensor(
            batch_rewards).unsqueeze(1).to(device)
        batch_next_states = torch.FloatTensor(batch_next_states).to(device)
        batch_dones = torch.FloatTensor(batch_dones).unsqueeze(1).to(device)

        current_q = online_net(batch_states).gather(1, batch_actions)
        next_q = target_net(batch_next_states).max(1)[0].unsqueeze(1)
        target_q = batch_rewards + DISCOUNT_FACTOR * next_q * (1 - batch_dones)

        loss = criterion(current_q, target_q.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if episode % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())

    if (episode + 1) % (NUM_EPISODES // 10) == 0:
        print(
            f"Episode {episode+1}/{NUM_EPISODES} - Total Reward: {total_reward:.2f} - Epsilon: {epsilon:.3f}")

training_time = time.time() - start_time
print(f"Training complete in {training_time:.2f} seconds.")

# --- Visualization of Learning Progress ---
chunk_size = 100
average_rewards = [np.mean(rewards_per_episode[i:i+chunk_size])
                   for i in range(0, len(rewards_per_episode)-chunk_size+1, chunk_size)]
episodes_chunked = [i * chunk_size for i in range(len(average_rewards))]
plt.figure(figsize=(10, 5))
plt.plot(episodes_chunked, average_rewards)
plt.xlabel("Episode")
plt.ylabel(f"Average Reward per {chunk_size} Episodes")
plt.title("DQN Learning on FrozenLake-v1 (Deterministic)")
plt.grid(True)
plt.show()

# --- Testing the Trained Agent ---
env_test = gym.make(ENV_NAME, is_slippery=False, render_mode='human')
num_test_episodes = 3
for episode in range(num_test_episodes):
    state, info = env_test.reset()
    state = int(state)
    state_vec = one_hot_state(state, num_states)
    print(f"\n--- Test Episode {episode+1} ---")
    done = False
    step_count = 0
    total_reward = 0
    while not done:
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = online_net(state_tensor)
        action = torch.argmax(q_values).item()
        next_state, reward, terminated, truncated, info = env_test.step(action)
        next_state = int(next_state)
        state_vec = one_hot_state(next_state, num_states)
        done = terminated or truncated
        total_reward += reward
        step_count += 1
        time.sleep(0.3)
    print(
        f"Test Episode {episode+1} finished in {step_count} steps. Total Reward: {total_reward}")
env_test.close()
env.close()
