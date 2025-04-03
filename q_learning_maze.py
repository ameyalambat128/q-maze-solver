import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time  # To add slight delay for visualization if needed

# --- Configuration & Hyperparameters ---
ENV_NAME = "FrozenLake-v1"
# Use is_slippery=False for a deterministic environment (easier to learn quickly)
# Use render_mode='human' to see the agent during testing, or 'rgb_array' for capturing frames
# Start with None for faster training
env = gym.make(ENV_NAME, is_slippery=False, render_mode=None)

# Q-learning hyperparameters
num_episodes = 10000       # Total training episodes
max_steps_per_episode = 100  # Avoid infinite loops in non-terminating scenarios

learning_rate = 0.1        # Alpha: How much new info overrides old
discount_factor = 0.99     # Gamma: Importance of future rewards

# Exploration-exploitation trade-off (Epsilon-greedy)
epsilon = 1.0               # Initial exploration rate
max_epsilon = 1.0           # Maximum exploration probability
min_epsilon = 0.01          # Minimum exploration probability
epsilon_decay_rate = 0.001  # How fast epsilon decreases (can be adjusted)
# Alternative decay: epsilon *= (1 - decay_rate) or exponential decay

# --- Q-Table Initialization (Person 2 Focus) ---
num_states = env.observation_space.n
num_actions = env.action_space.n
q_table = np.zeros((num_states, num_actions))  # Initialize Q-table with zeros

print(f"Environment: {ENV_NAME}")
print(f"Number of States: {num_states}")
print(f"Number of Actions: {num_actions}")
print(f"Initial Q-Table Shape: {q_table.shape}")

# --- Training Loop (Person 2 Focus, using Env from Person 1) ---
rewards_per_episode = []  # To track learning progress

print("\n--- Starting Training ---")
start_time = time.time()

for episode in range(num_episodes):
    # Reset environment for a new episode (Person 1 provides this interface)
    state, info = env.reset()
    state = int(state)  # Ensure state is an integer index

    terminated = False
    truncated = False
    current_episode_reward = 0

    for step in range(max_steps_per_episode):
        # 1. Choose Action (Epsilon-greedy)
        exploration_tradeoff = np.random.uniform(0, 1)

        if exploration_tradeoff < epsilon:
            action = env.action_space.sample()  # Explore: random action
        else:
            action = np.argmax(q_table[state, :])  # Exploit: best known action

        # 2. Take Action & Observe Outcome (Person 1 provides this interface)
        new_state, reward, terminated, truncated, info = env.step(action)
        new_state = int(new_state)  # Ensure state is an integer index

        # Adjust reward for FrozenLake (optional, helps learning faster)
        # Default: Goal=1, Hole=0, Move=0. Let's penalize falling slightly.
        if terminated and reward == 0.0:  # Fell in a hole
            reward = -0.1  # Small penalty
        # elif not terminated:
        #     reward = -0.01 # Tiny penalty for each step to encourage speed (optional)

        # 3. Update Q-Table (Bellman Equation)
        current_q = q_table[state, action]
        # Best Q-value for the next state
        max_future_q = np.max(q_table[new_state, :])

        # Q-learning formula
        new_q = current_q + learning_rate * \
            (reward + discount_factor * max_future_q - current_q)
        q_table[state, action] = new_q

        # Update state
        state = new_state
        current_episode_reward += reward

        # Check if episode finished
        if terminated or truncated:
            break

    # 4. Decay Epsilon (Exploration Rate) after each episode
    # Linear decay:
    epsilon = max(min_epsilon, epsilon - epsilon_decay_rate)
    # Exponential decay (alternative):
    # epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay_rate)) # Adjust decay rate if using this

    # Store rewards for plotting
    rewards_per_episode.append(current_episode_reward)

    # Print progress (optional)
    if (episode + 1) % (num_episodes // 10) == 0:
        print(f"Episode {episode + 1}/{num_episodes} | Epsilon: {epsilon:.3f}")

training_time = time.time() - start_time
print(f"--- Training Finished in {training_time:.2f} seconds ---")
print("\nFinal Q-Table:")
print(q_table)

# --- Visualization of Learning Progress (Both together) ---
# Calculate average rewards over chunks of episodes for smoother plot
chunk_size = 100
if len(rewards_per_episode) >= chunk_size:
    average_rewards = [np.mean(rewards_per_episode[i:i+chunk_size])
                       for i in range(0, len(rewards_per_episode) - chunk_size + 1, chunk_size)]
    episodes_chunked = [i * chunk_size for i in range(len(average_rewards))]

    plt.figure(figsize=(10, 5))
    plt.plot(episodes_chunked, average_rewards)
    plt.xlabel("Episodes")
    plt.ylabel(f"Average Reward per {chunk_size} Episodes")
    plt.title("Q-Learning Progress on FrozenLake-v1 (Deterministic)")
    plt.grid(True)
    plt.show()
else:
    print("\nNot enough episodes to plot meaningful average rewards.")


# --- Testing the Trained Agent (Optional Demo) ---
print("\n--- Testing Trained Agent ---")
env_test = gym.make(ENV_NAME, is_slippery=False,
                    render_mode='human')  # Use 'human' to see it

num_test_episodes = 3
for episode in range(num_test_episodes):
    state, info = env_test.reset()
    state = int(state)
    print(f"\n--- Test Episode {episode + 1} ---")
    time.sleep(1)  # Pause before start
    terminated = False
    truncated = False
    total_reward = 0
    step_count = 0

    while not terminated and not truncated:
        # Choose the best action (no exploration)
        action = np.argmax(q_table[state, :])

        new_state, reward, terminated, truncated, info = env_test.step(action)
        new_state = int(new_state)

        print(
            f"Step: {step_count+1}, State: {state}, Action: {action}, New State: {new_state}, Reward: {reward}")
        state = new_state
        total_reward += reward
        step_count += 1
        time.sleep(0.3)  # Slow down rendering

    print(f"Episode finished after {step_count} steps.")
    print(f"Total reward: {total_reward}")
    if reward == 1.0:
        print("Goal Reached!")
    elif terminated:
        print("Agent fell into a hole.")
    elif truncated:
        print("Episode truncated (max steps reached).")

    time.sleep(2)  # Pause between episodes

env.close()
env_test.close()
