import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from reward_callback import RewardCallback

env = Monitor(gym.make("CartPole-v1"))
callback = RewardCallback()
model = PPO("MlpPolicy",env,verbose=1)
model.learn(total_timesteps=100000,callback=callback)

np.save("rewards_original.npy",callback.episode_rewards)
model.save("ppo_original")
env.close()