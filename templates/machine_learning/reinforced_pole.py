import numpy as np

if not hasattr(np, 'bool8'):
  np.bool8 = np.bool_

import gymnasium as gym

env = gym.make('CartPole-v1', render_mode = "human")

for i_episode in range(10):
  observation, info = env.reset()

  for t in range(50):
    print(observation)

    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
      print('Episode finished after {} timestamps.'.format(t + 1))
      break

env.close()