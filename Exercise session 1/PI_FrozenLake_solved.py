# Copyright [2025] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the materials for EL2805 - Reinforcement Learning - Exercise Session 1 at KTH, Stockholm.

### IMPORT PACKAGES ###
import numpy as np                                                             # Import numpy for numerical operations and random number generation
import gymnasium as gym                                                        # Import gymnasium for creating and interacting with RL environments
from gymnasium.envs.toy_text.frozen_lake import generate_random_map            # Import a function to generate random maps for the FrozenLake environment
import matplotlib.pyplot as plt                                                # Import matplotlib for plotting



def policy_evaluation(env, num_states, num_actions, policy, gamma, max_iteration, tol):
    """
    Evaluate a given policy in a specified environment.

    Parameters:
    - env: the environment's transition probabilities and rewards structure
    - num_states: total number of states in the environment
    - num_actions: total number of actions possible
    - policy: the policy to evaluate (maps states to actions)
    - gamma: discount factor, controls the weight of future rewards
    - max_iteration: maximum number of iterations for evaluation
    - tol: tolerance for stopping criteria based on value changes

    Returns:
    - V: state-value function for the given policy
    """
    V = np.full(num_states, np.inf)                                            # Initialize value function for each state
    next_V = np.zeros(num_states)                                              # Placeholder for the updated value function
    i = 0                                                                      # Initialize iteration counter
    
    # Loop until max iterations or convergence based on tolerance
    while i <= max_iteration and np.linalg.norm(V - next_V) > tol:
        i += 1
        V = next_V.copy()
        # Update value function for each state based on the policy
        for s in range(num_states):
            env_data_s_pi = env[s][policy[s]]                                  # Get transition data for current state and pi(s)
            probabilities_s_pi = [row[0] for row in env_data_s_pi]             # Transition probabilities
            nextstates_s_pi = [row[1] for row in env_data_s_pi]                # Next states
            rewards_s_pi = [row[2] for row in env_data_s_pi]                   # Rewards
            dones_s_pi = [row[3] for row in env_data_s_pi]                     # Done flags
            
            # Tip: To inspect variables while the code is running, uncomment the line below to start an interactive debugger:
            # import pdb; pdb.set_trace()

            # TODO: Update value for state s based on Bellman equation
            next_V[s] = np.mean(rewards_s_pi) + gamma * np.dot(probabilities_s_pi, (1 - np.array(dones_s_pi)) * V[nextstates_s_pi])

    return V                                                                   # Return the evaluated value function



def policy_improvement(env_data, num_states, num_actions, V, gamma):
    """
    Improve the policy based on a given state-value function.

    Parameters:
    - env_data: environment's transition probabilities and rewards
    - num_states: total number of states
    - num_actions: total number of actions
    - V: state-value function from policy evaluation
    - gamma: discount factor

    Returns:
    - new_policy: improved policy derived from Q-values
    """
    Q = np.zeros([num_states, num_actions])                                    # Initialize Q-values
    
    # Calculate Q-values for each state-action pair
    for s in range(num_states):
        for a in range(num_actions):
            env_data_s_a = env_data[s][a]                                      # Get transition data for current state and action
            probabilities_s_a = [row[0] for row in env_data_s_a]               # Transition probabilities
            nextstates_s_a = [row[1] for row in env_data_s_a]                  # Next states
            rewards_s_a = [row[2] for row in env_data_s_a]                     # Rewards
            dones_s_a = [row[3] for row in env_data_s_a]                       # Done flags

            # TODO: Calculate Q-value for state-action pair using Bellman equation
            Q[s][a] = np.mean(rewards_s_a) + gamma * np.dot(probabilities_s_a,  (1 - np.array(dones_s_a)) * V[nextstates_s_a])
    
    new_policy = np.argmax(Q, axis=1)                                          # Derive policy by choosing action with highest Q-value    
    return new_policy                                                          # Return the improved policy



def policy_iteration(env_data, num_states, num_actions, gamma=0.9, max_PI_steps=200, max_eval_steps=100, tol_eval=1e-3):
    """
    Perform policy iteration to find the optimal policy.

    Parameters:
    - env_data: environment's transition probabilities and rewards
    - num_states: total number of states
    - num_actions: total number of actions
    - gamma: discount factor
    - max_iteration: maximum number of iterations
    - tol: tolerance for policy improvement convergence

    Returns:
    - V: final state-value function for the optimal policy
    - pi: optimal policy
    """
    V = np.zeros(num_states)                                                   # Initialize state-value function
    pi = np.zeros(num_states, dtype=int)                                       # Initialize a random policy
    V_history = [V.copy()]                                                     # Store initial value
    k = 0                                                                      # Initialize iteration counter
    
    # Iterate to improve policy until convergence or reaching maximum iterations
    next_pi = np.ones(num_states, dtype=int)
    while k <= max_PI_steps and np.any(next_pi != pi):
        k += 1
        pi = next_pi                                                           # Update current policy
        
        # TODO: Evaluate current policy using policy_evaluation function
        V = policy_evaluation(env_data, num_states, num_actions, pi, gamma, max_eval_steps, tol_eval)  
        
        # TODO: Do policy improvement using policy_improvement function
        next_pi = policy_improvement(env_data, num_states, num_actions, V, gamma) 
        
        V_history.append(V.copy())                                             # Track the value function after each iteration
        
    return V, pi, V_history                                                    # Return optimal state-value function and policy



def run_policy(env, policy, num_episodes, max_length):
    """
    Run a policy in the environment for multiple episodes and render it.

    Parameters:
    - env: the environment in which to run the policy
    - policy: the policy to execute
    - num_episodes: number of episodes to run
    - max_length: maximum length of each episode
    - render: whether to render the environment

    Returns:
    - total_episode_reward: array of total rewards for each episode
    """
    total_episode_reward = np.zeros(num_episodes)                              # Track reward per episode
    episode_lengths = np.ones(num_episodes)*max_length
    
    for episode in range(num_episodes):
        state = np.array(env.reset()[0])                                       # Reset environment for a new episode
        
        for t in range(max_length):

            action = policy[state]                                             # Choose action based on the policy for current state
            next_state, reward, done, trunc, _ = env.step(action)              # Take action and observe transitions
            
            total_episode_reward[episode] += reward                            # Accumulate reward for this episode
            state = next_state                                                 # Update state
            
            if done or trunc:
                episode_lengths[episode] = t
                break                                                          # End episode if terminal state (reached goal) 
                                                                               #                or truncation (reached max episode length)       
    return total_episode_reward, episode_lengths                               # Return total rewards for each episode



if __name__ == "__main__":
    
    # === Configuration flags ===
    RUN_PART_1 = True                                                          # Default FrozenLake (deterministic)
    RUN_PART_2 = False                                                         # Plot infinity norm differences
    RUN_PART_3 = False                                                         # Stochastic FrozenLake with histogram
    CUSTOM_MAP = False                                                         # Use a custom map
    RANDOM_MAP = False                                                         # Generate a random map
    # ===========================

    # --- Environment setup ---
    if CUSTOM_MAP:
        my_desc = ["SHHHH", "FFHFH", "FFFFH", "HFFFG"]
        env = gym.make('FrozenLake-v1', desc=my_desc, is_slippery=False, render_mode="human").unwrapped
    elif RANDOM_MAP:
        # Note: Computing the optimal policy on a 24x24 map may take several seconds
        env = gym.make('FrozenLake-v1', desc=generate_random_map(size=24), is_slippery=False, render_mode="human").unwrapped 
    else:
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False, render_mode="human").unwrapped


    # --- PART 1 ---
    if RUN_PART_1:
        V_star, pi_star, V_history = policy_iteration(env.P, env.observation_space.n, env.action_space.n, gamma=0.9)
        total_episode_reward, episode_lengths = run_policy(env, pi_star, num_episodes=1, max_length=50)
        print("Episode reward: %f" % total_episode_reward[0])                  


    # --- PART 2 ---
    if RUN_PART_2:
        infinity_norm_diffs = [np.max(np.abs(V - V_star)) for V in V_history]

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(infinity_norm_diffs, marker='o')
        plt.xlabel("Iteration (k)", size=18)
        plt.ylabel(r"$\Vert V^{\pi_k} - V^{\pi^\star} \Vert_\infty$", size=18)
        plt.title("Infinity Norm of Value Function Differences", size=18)

        plt.subplot(1, 2, 2)
        for s in range(len(V_star)):
            state_diffs = [V[s] - V_star[s] for V in V_history]
            plt.plot(state_diffs, marker='o', label=f"State {s}")
        plt.xlabel("Iteration (k)", size=18)
        plt.ylabel(r"$V^{\pi_k}(s) - V^{\pi^\star}(s)$", size=18)
        plt.title("Value Function Differences by State", size=18)
        plt.tight_layout()
        plt.show()


    # --- PART 3 ---
    if RUN_PART_3:
        env_stochastic = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True).unwrapped
        V_star, pi_star, V_history = policy_iteration(env_stochastic.P, env_stochastic.observation_space.n, env_stochastic.action_space.n, gamma=0.9)
        num_episodes = 100
        total_episode_reward, episode_lengths = run_policy(env_stochastic, pi_star, num_episodes, max_length=100)
        print("Reached goal in %.1f%% of simulated episodes." % (100*sum(total_episode_reward)/num_episodes))      

        plt.figure(figsize=(8, 6))
        plt.hist(episode_lengths, bins=20, color='skyblue', edgecolor='black')
        plt.xlabel("Episode Length", size=18)
        plt.ylabel("Frequency", size=18)
        plt.title("Histogram of Episode Lengths", size=18)
        plt.show()

        