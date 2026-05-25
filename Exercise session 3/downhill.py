# Copyright [2025] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the materials for EL2805 - Reinforcement Learning - Exercise Session 3 at KTH, Stockholm.

import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

# Environment parameters
TARGET_VELOCITY = 10
ACTION_ACCELERATE = 1
ACTION_DECELERATE = -1
GRAVITY_UPHILL = 0.5
GRAVITY_DOWNHILL = -0.5

# Simulation parameters
NUM_EPISODES_TEST = 50
MAX_STEPS = 10000
ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor
BATCH_SIZE = 10 
BUFFER_SIZE = 5000
SWITCH_INTERVAL = 2000  # Number of steps before switching tasks
NUM_RUNS = 25  # Number of independent training runs

# States are velocity values
velocity_states = np.arange(-10, 30)
actions = [ACTION_ACCELERATE, ACTION_DECELERATE]

# Helper function to discretize states
def get_state_index(velocity):
    return np.clip(int(velocity) + 10, 0, len(velocity_states) - 1)

# Reward function
def get_reward(velocity):
    return -abs(velocity - TARGET_VELOCITY)

# Transition function
def transition(velocity, action, scenario):
    gravity = GRAVITY_UPHILL if scenario == "uphill" else GRAVITY_DOWNHILL
    return velocity + action + gravity

# Train Q-learning with specified replay method
def train_q_learning(replay_method="correlated"):
    Q = np.zeros((len(velocity_states), len(actions)))
    replay_buffer = []

    velocity = random.uniform(5, 15)  # Random initial velocity
    for step in range(MAX_STEPS):
        # Determine scenario based on step
        scenario = "uphill" if (step // SWITCH_INTERVAL) % 2 == 0 else "downhill"
        state_idx = get_state_index(velocity)

        # Epsilon-greedy action selection
        if random.random() < 0.1:  # Exploration
            action = random.choice(actions)
        else:  # Exploitation
            action = actions[np.argmax(Q[state_idx])]

        # Environment transition
        next_velocity = transition(velocity, action, scenario)
        reward = get_reward(next_velocity)
        next_state_idx = get_state_index(next_velocity)

        # Store transition in buffer
        replay_buffer.append((state_idx, action, reward, next_state_idx))
        if len(replay_buffer) > BUFFER_SIZE:
            replay_buffer.pop(0)

        # Sample a batch for training
        if replay_method == "correlated":
                                                                               # TODO: add sequential sampling from the replay buffer
        else:
                                                                               # TODO: add random sampling from the replay buffer

        # Update Q-values for batch
        for s, a, r, s_next in batch:
            a_idx = actions.index(a)
            Q[s, a_idx] += ALPHA * (r + GAMMA * np.max(Q[s_next,:]) - Q[s, a_idx])

        velocity = next_velocity

    return Q

# Evaluate policy
def evaluate_policy(Q):
    rewards = []
    for scenario in ["uphill", "downhill"]:
        total_reward = 0
        velocity = random.uniform(8, 12)
        for step in range(NUM_EPISODES_TEST):
            state_idx = get_state_index(velocity)
            action = actions[np.argmax(Q[state_idx])]
            velocity = transition(velocity, action, scenario)
            total_reward += get_reward(velocity)
        rewards.append(total_reward)
    return rewards

# Train policies multiple times and average evaluations
def run_experiment():
    correlated_results = []
    randomized_results = []

    for i in tqdm(range(NUM_RUNS), desc="Training Progress"):
        Q_correlated = train_q_learning(replay_method="correlated")
        Q_randomized = train_q_learning(replay_method="randomized")

        correlated_rewards = evaluate_policy(Q_correlated)
        randomized_rewards = evaluate_policy(Q_randomized)

        correlated_results.append(correlated_rewards)
        randomized_results.append(randomized_rewards)
        
    return correlated_results, randomized_results


random.seed(27)

# Run the experiment
correlated_results, randomized_results = run_experiment()

# Plot results
scenarios = ["Uphill", "Downhill"]
x = np.arange(len(scenarios))
width = 0.35

q25_correlated = np.percentile(correlated_results, 25, axis=0)
q75_correlated = np.percentile(correlated_results, 75, axis=0)
q25_randomized = np.percentile(randomized_results, 25, axis=0)
q75_randomized = np.percentile(randomized_results, 75, axis=0)
median_corr = np.median(correlated_results, axis=0)
median_rand = np.median(randomized_results, axis=0)

plt.bar(x - width/2, median_corr, width, 
        yerr=[median_corr - q25_correlated, q75_correlated - median_corr],
        capsize=5, label="Correlated Replay")
plt.bar(x + width/2, median_rand, width,
        yerr=[median_rand - q25_randomized, q75_randomized - median_rand],
        capsize=5, label="Randomized Replay")

plt.xticks(x, scenarios)
plt.ylabel("Average Total Reward")
plt.title("Policy Performance Comparison (Correlated vs. Randomized Replay)")
plt.legend()
plt.show()

