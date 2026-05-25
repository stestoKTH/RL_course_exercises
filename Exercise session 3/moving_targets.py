# Copyright [2025] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the materials for EL2805 - Reinforcement Learning - Exercise Session 3 at KTH, Stockholm.

import numpy as np
import matplotlib.pyplot as plt

# ---------------- Part 1: Unstable system (Exercise 6) ----------------

A = 0.8 * np.array([[0.0, 1.0],                                                # Case 1
                    [-1.0, 0.0]])
# A = np.array([[0.0, 1.0],                                                    # Case 2
#                     [-1.0, 0.0]])
# A = 1.2 * np.array([[0.0, 1.0],                                              # Case 3
#                     [-1.0, 0.0]])

b = np.array([0.5, 0.8])

T = 500                                                                        # number of update iterations
theta = np.array([1.0, 0.0])                                                   # initial parameters estimate
traj_no_target = []                                                            # a list of parameters encountered during iterations

# Run unstable updates
theta_nt = theta.copy()
for t in range(T):
    theta_nt =                                                                 # TODO: write parameter update
    traj_no_target.append(theta_nt.copy())

traj_no_target = np.array(traj_no_target)
true_solution =                                                                # TODO: find fixed point solution

print("Final parameters without target network:", traj_no_target[-1])
print("True solution (fixed point):", true_solution)

plt.figure()
plt.plot(traj_no_target[:,0], traj_no_target[:,1])
plt.title("Unstable system WITHOUT target network")
plt.xlabel("theta1")
plt.ylabel("theta2")
plt.axis("equal")


# ---------------- Part 2: Stabilized system with target networks (Exercise 7) ----------------
alpha = 0.1                                                                    # online update rate
K = 10                                                                         # hard target network refresh period
tau = 0.05                                                                     # Polyak averaging parameter

# --- Hard target network updates ---
theta_t = theta.copy()
theta_target = theta.copy()                                                    # target parameters
traj_hard_target = []                                                          # a list of estimated parameters with K-step updated target networks

for t in range(T):
    # Online update toward target
    theta_t =                                                                  # TODO: write parameter update with fixed target
    traj_hard_target.append(theta_t.copy())
    
    # Refresh target network every K steps
    if t % K == 0:
        theta_target =                                                         # TODO: update the target network

traj_hard_target = np.array(traj_hard_target)

# --- Polyak averaging (soft target network) ---
theta_t = theta.copy()
theta_target = theta.copy()
traj_polyak = []                                                               # a list of estimated parameters with Polyak averaged target networks

for t in range(T):
    # Online update toward target
    theta_t =                                                                  # TODO: write parameter update with fixed target
    traj_polyak.append(theta_t.copy())
    
    # Soft update (Polyak averaging)
    theta_target =                                                             # TODO: update the target network with Polyak averaging

traj_polyak = np.array(traj_polyak)

print("Final parameters with hard target network:", traj_hard_target[-1])
print("Final parameters with Polyak averaging:", traj_polyak[-1])

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(traj_hard_target[:,0], traj_hard_target[:,1])
plt.title("Stabilized system WITH hard target network")
plt.xlabel("theta1")
plt.ylabel("theta2")
plt.axis("equal")

plt.subplot(1,2,2)
plt.plot(traj_polyak[:,0], traj_polyak[:,1])
plt.title("Stabilized system WITH Polyak averaging")
plt.xlabel("theta1")
plt.ylabel("theta2")
plt.axis("equal")

plt.tight_layout()
plt.show()

