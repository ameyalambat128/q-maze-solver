import gym
import numpy as np

# Add compatibility for newer NumPy versions
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")

obs = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # Take a random action
    obs, reward, done, truncated, info = env.step(action)
    env.render()

print("Episode finished. Reward:", reward)
env.close()
