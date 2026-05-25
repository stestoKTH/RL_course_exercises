# Copyright [2025] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the materials for EL2805 - Reinforcement Learning - Exercise Session 3 at KTH, Stockholm.

import numpy as np
import gymnasium as gym

# Parameters for Q-learning and SARSA
EPSILON = 0.1
ALPHA = 0.5
GAMMA = 1

# Custom reward wrapper for FrozenLake
class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        next_state, reward, done, truncated, info = self.env.step(action)
        # Modify reward based on next state
        if next_state == 7:  # Falling into a hole
            reward = -1000
        else:
            reward = -1  # Default reward
        return next_state, reward, done, truncated, info


# Create environment with custom rewards
my_desc = ["FFF", "FFF", "SHG"]
env = CustomRewardWrapper(gym.make('FrozenLake-v1', desc=my_desc, is_slippery=False).unwrapped)

# Initialize Q-values
q_sarsa = np.zeros((env.observation_space.n, env.action_space.n))
q_q_learning = np.copy(q_sarsa)

# Choose action based on epsilon-greedy policy
def choose_action(state, q_value):
    if np.random.binomial(1, EPSILON) == 1:
        return np.random.choice(env.action_space.n)  # Random action
    else:
        values_ = q_value[state, :]
        return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

# SARSA update rule
def sarsa(q_value, step_size=ALPHA):
    state = env.reset()[0]  # Initial state
    action = choose_action(state, q_value)
    done, truncated = False, False
    while not (done or truncated):
        next_state, reward, done, truncated, _ = env.step(action)
        next_action = choose_action(next_state, q_value)
        q_value[state, action] += step_size * (reward + GAMMA * q_value[next_state, next_action] - q_value[state, action])
        state, action = next_state, next_action
    return q_value

# Q-Learning update rule
def q_learning(q_value, step_size=ALPHA):
    state = env.reset()[0]  # Initial state
    done, truncated = False, False
    while not (done or truncated):
        action = choose_action(state, q_value)
        next_state, reward, done, truncated, _ = env.step(action)
        q_value[state, action] += step_size * (reward + GAMMA * np.max(q_value[next_state, :]) - q_value[state, action])
        state = next_state
    return q_value

# Print optimal policy
def print_optimal_policy(q_value):
    optimal_policy = []
    for i in range(0, 3):
        optimal_policy.append([])
        for j in range(0, 3):
            if [i, j] == [2, 2]:
                optimal_policy[-1].append('G')
                continue
            best_action = np.argmax(q_value[i * 3 + j, :])
            if best_action == 0:
                optimal_policy[-1].append('L')
            elif best_action == 1:
                optimal_policy[-1].append('D')
            elif best_action == 2:
                optimal_policy[-1].append('R')
            elif best_action == 3:
                optimal_policy[-1].append('U')
    for row in optimal_policy:
        print(row)

# Train both SARSA and Q-learning for multiple episodes
episodes = 10000
for i in range(episodes):
    q_sarsa = sarsa(q_sarsa)
    q_q_learning = q_learning(q_q_learning)

# Display learned policies
print('SARSA Optimal Policy:')
print_optimal_policy(q_sarsa)
print('Q-Learning Optimal Policy:')
print_optimal_policy(q_q_learning)

# Render policy on the environment
def render_policy(q_value, title):
    env = CustomRewardWrapper(gym.make('FrozenLake-v1', desc=my_desc, is_slippery=False, render_mode="human").unwrapped)
    state = env.reset()[0]
    done, truncated = False, False
    while not (done or truncated):
        action = np.argmax(q_value[state, :])  # Follow policy
        next_state, _, done, truncated, _ = env.step(action)
        state = next_state
    env.close()

# Render the policies
print("\nRendering SARSA Policy...")
render_policy(q_sarsa, "SARSA Policy")

print("\nRendering Q-Learning Policy...")
render_policy(q_q_learning, "Q-Learning Policy")
