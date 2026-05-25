# Copyright [2025] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the materials for EL2805 - Reinforcement Learning - Exercise Session 4 at KTH, Stockholm.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import time


# Hyperparameters for the REINFORCE algorithm
learning_rate = 0.01                                                           # Learning rate for the optimizer
hidden_units = 32                                                              # Number of hidden units in the policy network
num_iterations = 51                                                            # Number of training iterations
N_env = 10

# Environment setup: Create the CartPole environment
def make_env():
    return lambda: gym.make('CartPole-v1')

# Create N parallel environments
envs = gym.vector.SyncVectorEnv([make_env() for _ in range(N_env)])
state_shape = envs.single_observation_space.shape[0]
num_actions = envs.single_action_space.n


# Define the policy network: a simple neural network to approximate the policy
class PolicyNetwork(nn.Module):
    def __init__(self, state_shape, num_actions):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_shape, hidden_units)                        # First hidden layer
        self.fc2 = nn.Linear(hidden_units, num_actions)                        # Output layer (logits for actions)
        self.softmax = nn.Softmax(dim=-1)                                      # Softmax to convert logits to probabilities

    def forward(self, state):
        # Forward pass through the network
        x = torch.relu(self.fc1(state))                                        # Apply ReLU activation
        action_probs = self.softmax(self.fc2(x))                               # Get action probabilities via Softmax
        return action_probs

# Function to compute normalized cost-to-go based on different methods
def find_returns(episode_rewards_all, method="vanilla"):

    all_returns = []
    returns_mean = []

    # Convert each episode to a numpy array
    if method == "remove_baseline" or method == "both":
        # Pad episodes with zeroes for unequal lengths
        max_len = max(len(ep) for ep in episode_rewards_all)
        padded_rewards = np.full((len(episode_rewards_all), max_len), 0, dtype=np.float32)
        for i, ep in enumerate(episode_rewards_all):
            padded_rewards[i, :len(ep)] = ep
            
        baseline = np.mean(padded_rewards, axis=0)

    for episode_rewards in episode_rewards_all:
        n = len(episode_rewards)
        total_rewards = np.zeros_like(episode_rewards, dtype=np.float32)

        if method == "vanilla":
            total_rewards =                                                    # TODO: find vanilla return for current episode
            returns_mean.append(total_rewards[0])

        elif method == "reward_to_go":                                         # TODO: find cost-to-go for current episode
            reward_to_go = 0.0
            for i in reversed(range(n)):
                reward_to_go = 
                total_rewards[i] = 
            returns_mean.append(total_rewards[0])

        elif method == "remove_baseline":                                      # TODO: find return with subtracted baseline
            total_rewards =
            returns_mean.append(total_rewards[0])
            total_rewards -=

        elif method == "both":                                                 # TODO: find cost-to-go with subtracted (temporal) baseline
            reward_to_go = 0.0
            for i in reversed(range(n)):
                reward_to_go =
                if i==0: returns_mean.append(reward_to_go)
                total_rewards[i] =

        all_returns.append(total_rewards)
    
    # Flatten to a single array for all environments
    return np.concatenate(all_returns), returns_mean



# Training function using REINFORCE algorithm with different reward discounting methods
def train(method="vanilla"):
    # Initialize policy network and optimizer
    policy_net = PolicyNetwork(state_shape, num_actions)
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    
    # Training loop
    for iteration in range(num_iterations):
        episode_states = [[] for _ in range(N_env)]
        episode_actions = [[] for _ in range(N_env)]
        episode_rewards = [[] for _ in range(N_env)]
        states = envs.reset()[0]  # returns (obs, info)
        stop_flags = np.zeros(N_env, dtype=bool)
        
        # Collect data from an episode
        while not all(stop_flags):
            states_tensor = torch.FloatTensor(states)                          # shape (N_env, state_dim)
            action_probs = policy_net(states_tensor).detach().numpy()          # shape (N_env, num_actions)
            actions = [np.random.choice(num_actions, p=action_probs[i]) for i in range(N_env)]
            next_states, rewards, dones, truncs, infos = envs.step(actions)
    
            # Store data
            for i in range(N_env):
                if not stop_flags[i]:  # Only store if env not done
                    one_hot = np.zeros(num_actions, dtype=np.float32)
                    one_hot[actions[i]] = 1.0
                    episode_states[i].append(states[i])
                    episode_actions[i].append(one_hot)
                    episode_rewards[i].append(rewards[i])
                    if dones[i] or truncs[i]:
                        stop_flags[i] = True
            
            states = next_states
    
        # Compute causal and normalized rewards for the episode
        tmp_returns, returns_mean = find_returns(episode_rewards, method)
    
        # Convert data to tensors for training
        flat_states = [s for env_states in episode_states for s in env_states]
        flat_actions = [a for env_actions in episode_actions for a in env_actions]
        
        states_tensor = torch.FloatTensor(np.array(flat_states, dtype=np.float32))
        actions_tensor = torch.FloatTensor(np.array(flat_actions, dtype=np.float32))
        returns_tensor = torch.FloatTensor(tmp_returns)
    
        # Compute loss (negative log probability of taken actions weighted by rewards)
        action_probs = policy_net(states_tensor)                               # Get action probabilities from the network
        log_probs = torch.log(torch.sum(action_probs * actions_tensor, dim=1)) # Log probability of taken actions
        loss = -torch.mean(log_probs * returns_tensor)                         # Compute the loss (negative log-likelihood)
    
        # Optimize the policy network
        optimizer.zero_grad()                                                  # Clear previous gradients
        loss.backward()                                                        # Backpropagate the loss
        optimizer.step()                                                       # Update the network weights
     
        # Log progress every 10 iterations
        if iteration % 10 == 0:
            print(f"Iteration: {iteration}, Return: {np.array(returns_mean).mean():.2f}")
    
    return policy_net                                                          # Return the trained policy network

# Simulation function to run the trained policy and visualize the agent interacting with the environment
def simulate(policy_net):
    # env = gym.make('CartPole-v1', render_mode="human")                         # Create environment with rendering enabled
    env = gym.make('CartPole-v1')                                              # Create environment with rendering disabled
    done, truncated = False, False
    state, _ = env.reset()                                                     # Reset environment
    total_reward = 0.0
    # Run the simulation
    while not (done or truncated):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)                   # Convert state to tensor (with batch dimension)
        action_probs = policy_net(state_tensor).detach().numpy().ravel()       # Get action probabilities
        action = np.random.choice(num_actions, p=action_probs)                 # Sample an action based on probabilities
        next_state, reward, done, truncated, _ = env.step(action)              # Take step in environment
        state = next_state                                                     # Move to the next state
        total_reward += reward
        # time.sleep(0.1)                                                        # Add a small delay for visualization
    env.close()                                                                # Close the environment after the simulation
    return total_reward

# Run the different training scenarios

print("Training Vanilla REINFORCE...")
policy_net_vanilla = train(method="vanilla")  
total_reward1 = simulate(policy_net_vanilla)  
print(f"Total return: {total_reward1:.2f}")

print("Training REINFORCE with reward_to_go...")
policy_net_togo = train(method="reward_to_go")  
total_reward2 = simulate(policy_net_togo)  
print(f"Total return: {total_reward2:.2f}")

print("Training REINFORCE with baseline...")
policy_net_togo = train(method="remove_baseline") 
total_reward3 = simulate(policy_net_togo)  
print(f"Total return: {total_reward3:.2f}")

print("Training REINFORCE with reward-to-go and baseline...")
policy_net_both = train(method="both")
total_reward4 = simulate(policy_net_both) 
print(f"Total return: {total_reward4:.2f}")
