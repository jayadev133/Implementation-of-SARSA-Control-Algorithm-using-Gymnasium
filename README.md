# Implementation-of-SARSA-Control-Algorithm-using-Gymnasium

## Aim

To implement the **SARSA control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an action-value function that helps the agent select better actions for reaching the goal state while avoiding holes.

---

## Problem Statement
The problem is to implement the SARSA algorithm for training an agent in the FrozenLake environment. The agent learns the best actions using an epsilon-greedy policy and updates Q-values based on the action actually selected. The goal is to maximize cumulative rewards and successfully reach the destination.


## Software Requirements
Python 3.x
Jupyter Notebook 
NumPy ,Gymnasium (FrozenLake) , Matplotlib


## Environment Description
The FrozenLake environment is a grid-based environment where an agent moves across frozen tiles to reach the goal. The agent can move up, down, left, or right, while avoiding holes that end the episode. The agent receives a reward of 1 for reaching the goal and 0 for other movements, allowing SARSA to learn the optimal path through repeated interaction.


## Theory

SARSA stands for:

$$
S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}
$$

It updates the Q-value using the action actually selected in the next state.

The SARSA update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $A_{t+1}$ | Next action selected using the current policy |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |

---

## Epsilon-Greedy Policy

SARSA uses an epsilon-greedy policy for action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_a Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---


## Algorithm
Initialize the Q-table with zeros for all states and actions.

Initialize parameters such as learning rate \(\alpha\), discount factor \(\gamma\), and exploration rate \(\epsilon\).

Reset the environment and select an initial action using the epsilon-greedy policy.

Take the selected action, observe the reward and next state, and select the next action using the same policy.

Update the Q-value using the SARSA update rule: \(Q(s,a) \leftarrow Q(s,a)+\alpha[r+\gamma Q(s',a')-Q(s,a)]\).

Repeat the process until the episode terminates, while gradually decreasing \(\epsilon\).

Evaluate the learned Q-table to obtain the optimal policy and analyze the rewards over episodes.


## Python Program

```python

# -------------------------------------------------
# SARSA Training
# -------------------------------------------------

episode_rewards = []

for episode in range(num_episodes):

    state, info = env.reset()

    # Choose the first action using epsilon-greedy
    action = epsilon_greedy_action(state, epsilon)

    total_reward = 0

    for step in range(max_steps_per_episode):

        # Take action
        next_state, reward, terminated, truncated, info = env.step(action)

        # Choose next action using the current policy
        next_action = epsilon_greedy_action(next_state, epsilon)

        # SARSA update
        if terminated or truncated:
            target = reward
        else:
            target = reward + gamma * Q[next_state, next_action]

        Q[state, action] = Q[state, action] + alpha * (
            target - Q[state, action]
        )

        # Move to next state and action
        state = next_state
        action = next_action

        total_reward += reward

        if terminated or truncated:
            break

    # Store episode reward
    episode_rewards.append(total_reward)

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)


# -------------------------------------------------
# Extract Value Function and Learned Policy
# -------------------------------------------------

state_values = np.max(Q, axis=1)

learned_policy = np.argmax(Q, axis=1)

print("Training completed.")




```
---

## Output


Final Q-table:

<img width="381" height="261" alt="image" src="https://github.com/user-attachments/assets/2d119b6f-7666-4097-9b35-b17cbeafd321" />





Estimated State-Value Function:


<img width="252" height="81" alt="image" src="https://github.com/user-attachments/assets/21fdbdf0-6304-47f1-aba9-9da26a3bbeb0" />



Learned Policy:

<img width="235" height="91" alt="image" src="https://github.com/user-attachments/assets/f26fcbf1-55a6-4e0d-9d54-088c87839d62" />




Average reward over last 1000 episodes: 


<img width="438" height="25" alt="image" src="https://github.com/user-attachments/assets/5cd4c9f2-7e71-4b46-92e4-df13870930f2" />



## Result
```text

The SARSA algorithm successfully trained the agent to learn an effective policy for navigating the FrozenLake environment. The learned Q-values and rewards demonstrate the agent’s ability to improve its actions and reach the goal through repeated learning.

```

---

## Inference
```text
The agent gradually learns from its actions and improves its decision-making through repeated interaction with the environment.

SARSA shows that using the actual next action helps the agent learn a suitable path to the goal while balancing exploration and exploitation.


```
---

