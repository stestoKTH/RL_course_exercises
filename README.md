# Exercises in Reinforcement Learning @ KTH (2025)

This repository contains the exercise sessions developed for **EL2805 Reinforcement Learning** at KTH Royal Institute of Technology (2025 edition). The goal is to complement the lectures with hands-on implementation, intuition-building, and theoretical insights into reinforcement learning.

If you find any mistakes or issues in the material, feel free to contact me.

---

## Exercise Sessions

### 1. Markov Decision Processes & Dynamic Programming
This session is split into two parts. The first part focuses on why policy iteration works, with a study of the theoretical convergence properties of policy iteration and practical experiments in the FrozenLake environment.  

The second part explores why supervised learning approaches fail in sequential decision-making tasks, motivating the need for reinforcement learning instead of pure behavioral cloning.

---

### 2. Monte Carlo, Temporal Difference Learning & Eligibility Traces
This session compares Monte Carlo methods with temporal difference learning. It then introduces eligibility traces as a bridge between the two approaches, helping to unify different temporal credit assignment mechanisms.

---

### 3. From Q-learning to Deep Q-Networks
We begin with SARSA and Q-learning, and then analyze key challenges that arise when combining Q-learning with neural networks: maximization bias, data correlations, experience replay, and moving targets. The session concludes with an implementation of Deep Q-Networks (DQN).

---

### 4. Policy Gradient Methods: REINFORCE & Trust Region Policy Optimization
This session introduces policy gradient methods starting from REINFORCE, highlighting the causality principle and variance reduction via baseline subtraction. We then derive Trust Region Policy Optimization (TRPO) and discuss its theoretical guarantees.

---

### 5. Soft Actor-Critic and Entropy-Regularized Reinforcement Learning
This session focuses on exploration in reinforcement learning. We introduce entropy as an exploration mechanism, and study Soft Actor-Critic (SAC) as an actor-critic method that naturally incorporates entropy maximization.

---

### 6. Large Language Model Alignment: RLHF & Direct Preference Optimization
This session connects reinforcement learning to modern language model training. We discuss Reinforcement Learning from Human Feedback (RLHF) using Proximal Policy Optimization (PPO), and compare it with Direct Preference Optimization (DPO) as a simpler alternative that avoids explicit RL loops.
