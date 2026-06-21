import gymnasium as gym
env = gym.make("CartPole-v1", render_mode="human")

for episode in range(1, 50):
    score=0
    state, info=env.reset()
    done=False
    while not done:
        action=env.action_space.sample()
        n_state, reward, terminated, truncated, info = env.step(action)
        done=terminated or truncated
        score += reward

    print('Episode: ', episode, 'Score: ', score)

env.close()
