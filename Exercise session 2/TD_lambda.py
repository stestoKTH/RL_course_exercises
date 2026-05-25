# Copyright [2025] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the materials for EL2805 - Reinforcement Learning - Exercise Session 2 at KTH, Stockholm.

### IMPORT PACKAGES ###
import gymnasium as gym                                                        # Import the gymnasium library for creating and interacting with reinforcement learning environments
import numpy as np                                                             # Import numpy for numerical computations and managing arrays
import matplotlib.pyplot as plt                                                # Import matplotlib for creating visualizations
import matplotlib.colors as mcolors                                            # Import color and shape modules from matplotlib for customized plotting
from gymnasium.envs.toy_text.frozen_lake import generate_random_map            # Import a function to generate a random map layout for the FrozenLake environment
from PI_FrozenLake_solved import policy_iteration                              # Import function from Exercise 1 to solve the FrozenLake environment



def td_lambda(env, policy, lambd=0.9, alpha=0.1, gamma=0.9, num_episodes=1, epsilon=0.0):
    """
    Evaluate a fixed policy using TD(lambda) with eligibility traces.

    Parameters
    ----------
    env           : Environment with discrete states and actions
    policy        : Fixed policy mapping each state to an action
    lambd         : Trace decay rate (0 <= lambda <= 1)
    alpha         : Learning rate
    gamma         : Discount factor
    num_episodes  : Number of episodes to run
    epsilon       : Exploration probability; 0 means follow policy strictly

    Returns
    -------
    V            : Estimated state-value function
    V_start      : Value of the initial state after each episode
    """
    V = np.zeros(env.observation_space.n)                                      # Initialize value function for all states
    E = np.zeros(env.observation_space.n)                                      # Initialize eligibility traces for all states
    V_start = []

    # Run TD(lambda) algorithm for the specified number of episodes
    for episode in range(num_episodes):
        state, _ = env.reset()                                                 # Reset the environment at the beginning of each episode
        done = False                                                           # Flag for episode termination
        terminated = False                                                     # Flag for truncation
        E.fill(0)                                                              # Reset eligibility traces at the start of each episode

        # Loop until the episode ends (either done or terminated)
        while not (done or terminated):
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = policy[state]                                         # Choose action based on the policy
                                      
            next_state, reward, done, terminated, _ = env.step(action)         # Take action and observe result

            # Calculate TD error based on the reward and predicted next state value
            td_error =                                                         # TODO
            E[state] =                                                         # TODO: Increment eligibility trace for the current state

            # Update value function and eligibility traces for all states
            V =                                                                # TODO: Update state value with TD error and trace
            E =                                                                # TODO: Apply decay to eligibility traces
            state =                                                            # TODO: Move to the next state

        V_start.append(V[0])
    
    return V, V_start                                                          # Return final value function and value of initial state




def plot_updates(env, updates, lambd):
    """
    Visualize updates made to each state by TD(lambda).

    Parameters
    ----------
    env     : Environment object (FrozenLake) containing the lake layout
    updates : Array of update magnitudes for each state
    lambd   : Lambda parameter used in TD(lambda), shown in the title
    """
    lake_size = int(np.sqrt(env.observation_space.n))                          # Calculate the lake grid size
    lake = np.array(env.desc, dtype=str).reshape(lake_size, lake_size)         # Obtain lake layout

    # Create a numerical grid for lake layout (S=Start, G=Goal, H=Hole, F=Frozen)
    numerical_lake = np.zeros(lake.shape, dtype=int)
    numerical_lake[lake == 'S'] = 0                                            # Start state
    numerical_lake[lake == 'G'] = 1                                            # Goal state
    numerical_lake[lake == 'H'] = 2                                            # Hole
    numerical_lake[lake == 'F'] = 3                                            # Frozen state

    updates = updates.reshape(lake_size, lake_size)                            # Reshape updates to match lake grid
    
    fig, ax = plt.subplots()                                                   # Initialize plot
    cmap = mcolors.ListedColormap(['orange', 'green', 'blue', 'lightblue'])    # Color map for lake
    ax.imshow(numerical_lake, cmap=cmap, vmin=0, vmax=3)                       # Display lake layout with color map

    # Plot each update value on the lake with variable opacity
    for (i, j), update_value in np.ndenumerate(updates):
        if abs(update_value) > 0:
            ax.text(j, i, f'{update_value:.2f}', ha='center', va='center', color='brown')  # Display update
    
    ax.set_xticks([])                                                          # Remove x-axis ticks
    ax.set_yticks([])                                                          # Remove y-axis ticks
    plt.title(f"State Updates in TD($\\lambda={lambd}$)")                      # Add title
    plt.show()                                                                 # Display the plot





if __name__ == "__main__":
    
    # === Configuration flags ===
    RUN_PART_1 = True                                                        
    RUN_PART_2 = True                                                         
    RUN_PART_3 = True                                                         
    # ===========================

    # Create an 8x8 FrozenLake environment with a non-slippery surface 
    # Note: if reward_schedule does not work, try updating gymnasium to a newever version!
    env = gym.make('FrozenLake-v1', desc=generate_random_map(size=8), is_slippery=False, reward_schedule=(10,-1,-0.1), render_mode="rgb_array").unwrapped

    # Compute optimal policy using policy iteration for an 8x8 FrozenLake environment
    _, policy, _ = policy_iteration(env.P, env.observation_space.n, env.action_space.n, gamma=0.95, max_PI_steps=30, tol_eval=1e-3)

    # --- PART 1 ---
    if RUN_PART_1:

        # Run TD(lambda) for different lambda values and visualize updates
        for lambd in [0.0, 0.5, 0.99]:
            V, _ = td_lambda(env, policy, lambd=lambd, alpha=1, gamma=0.95, num_episodes=1)
            # V, _ = td_lambda(env, policy, lambd=lambd, alpha=1, gamma=0.95, num_episodes=100)
            plot_updates(env, V, lambd)                                        # Plot updates for each state       
            
            
    # --- PART 2 ---
    if RUN_PART_2:

        num_episodes = 20
        lambda_arr = [0.0, 0.2, 0.4, 0.6, 0.8, 0.99]
        costs = []
        for lambd in lambda_arr:
            _, V_start = td_lambda(env, policy, lambd=lambd, alpha=1.0, gamma=0.95, num_episodes=num_episodes)
            # _, V_start = td_lambda(env, policy, lambd=lambd, alpha=1.0, gamma=0.95, num_episodes=num_episodes, epsilon=0.1)
            costs.append(V_start) 

        # Plot
        plt.figure(figsize=(5, 3.5))
        colors = plt.cm.tab10(np.linspace(0, 1, len(lambda_arr)))        
        for i, lam in enumerate(lambda_arr):
            plt.plot(np.arange(num_episodes), costs[i], color=colors[i], label=f"λ={lam:.2f}", linewidth=2)
        
        plt.title("FrozenLake learning curves (at γ=0.95)")
        plt.xlabel("number of collected episodes")
        plt.ylabel("V(start)")
        plt.legend(frameon=False)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()


    # --- PART 3 ---
    if RUN_PART_3:
        # Example grid data (replace with your actual results)
        gamma_values = np.array([0.90, 0.92, 0.94, 0.96, 0.98, 0.99])
        lambda_values = np.array([0.9, 0.92, 0.94, 0.96, 0.98, 0.99])
        performance = np.zeros([len(gamma_values),len(lambda_values)])

        for cnt_g, gamma in enumerate(gamma_values):
            for cnt_l, lambd in enumerate(lambda_values):
                _, V_start = td_lambda(env, policy, lambd=lambd, alpha=1, gamma=gamma, num_episodes=1)
                # _, V_start = td_lambda(env, policy, lambd=lambd, alpha=1.0, gamma=gamma, num_episodes=10, epsilon=0.1)
                performance[cnt_g, cnt_l] = V_start[-1]
            
        plt.figure(figsize=(5, 3.5))
        plt.imshow(performance, cmap='gray_r', aspect='auto', origin='lower')
        
        plt.title("FrozenLake performance after 1 episode")
        plt.xlabel(r'$\lambda$')
        plt.ylabel(r'$\gamma$')
        
        # Tick positions
        plt.xticks(ticks=np.arange(len(lambda_values)), labels=[f"{x:.2f}" for x in lambda_values])
        plt.yticks(ticks=np.arange(len(gamma_values)), labels=[f"{y:.2f}" for y in gamma_values])
        
        plt.tight_layout()
        plt.show()
  