import matplotlib.pyplot as plt
import numpy as np
from gridworld_env import GridWorldEnv
from dqn_agent import DQNAgent

env = GridWorldEnv()
agent = DQNAgent(state_dim=2, action_dim=4)

episodes = 500
max_steps = 200
rewards_history = []

for episode in range(episodes):
    state = env.reset()
    total_reward = 0

    for step in range(max_steps):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)

        agent.store(state, action, reward, next_state, done)
        agent.train()

        state = next_state
        total_reward += reward

        if done:
            break

    rewards_history.append(total_reward)

    if episode % 50 == 0:
        print(f"Episode {episode}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

# Smooth rewards
window = 20
smoothed_rewards = np.convolve(
    rewards_history, np.ones(window) / window, mode="valid"
)

plt.plot(smoothed_rewards)
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("DQN Learning Curve (Smoothed)")
plt.savefig("results/rewards.png")
plt.show()
