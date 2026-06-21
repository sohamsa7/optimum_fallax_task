import gymnasium as gym
from stable_baselines3 import PPO
env = gym.make("CartPole-v1", render_mode='human')
model = PPO.load("ppo_cartpole")
scores = []
for episode in range(20):
    obs, info = env.reset()
    done = False
    score = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        score += reward


    scores.append(score)
    print(f'Episode: {episode}, score: {score}')


print("Average Reward:", sum(scores)/len(scores))
