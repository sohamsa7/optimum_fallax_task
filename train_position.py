import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from reward_callback import RewardCallback

class PositionRewardWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = 1 + abs(obs[0])
        return obs, reward, terminated, truncated, info

env = Monitor(PositionRewardWrapper(gym.make("CartPole-v1")))
callback = RewardCallback()
model = PPO("MlpPolicy",env,verbose=1)

model.learn(total_timesteps=100000,callback=callback)

np.save("rewards_position.npy",callback.episode_rewards)
model.save("ppo_position")
env.close()