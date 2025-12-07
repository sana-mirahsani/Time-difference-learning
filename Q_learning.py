# ==========================================================
# This file includes the Q-learning alogorithm
# ==========================================================

# =============================================================================
# 0. Libraries
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import random
# =============================================================================
# 1. Q_Learning
# =============================================================================
def Q_learning_func(states, actions, discount_value, select_action_strategy, max_episodes=100, stop_condtion=None, obj_env=None):
    """
    Q_learning algorithm.

    Args:
        states : set
            Set of states.
        actions : set
            Set of actions.
        discount_value : float
            A float number.
    Returns:
        Q_table : numpy array 2D
            The Q value for all states and actions pairs.
    """
    
    # Initializing
    Q_hat = np.zeros((len(states),len(actions)), dtype=int)
    num_visit = np.zeros((len(states),len(actions)), dtype=int)
    epsilon = 1.0           # initial epsilon
    epsilon_decay = 0.99     # decay factor
    epsilon_min = 0.01       # optional lower bound
    epsilon_values = [] # to check the condition of epsilon later

    # outer loop
    for _ in range(max_episodes):
        # choose the start state randomly
        current_state = random.choice(states)
        done = False

        while not done:
            # choose an action by one of the strategies
            s_idx = states.index(current_state)
            action = select_action_strategy(s_idx, Q_hat,actions, epsilon) 

            # observe st+1 and rt
            next_state, reward, done = obj_env.step(current_state, action) 

            # find index of state and action
            a_idx = actions.index(action)
            next_s_idx = states.index(next_state)

            # calculate the TD error
            TD_error = reward + (discount_value * (np.max(Q_hat[next_s_idx,:]))) - Q_hat[s_idx, a_idx]

            # calculate learning step
            learning_step = 1/(num_visit(s_idx,a_idx) + 1)

            # update Q_hat
            Q_hat[s_idx, a_idx] = Q_hat[s_idx, a_idx] + learning_step * (TD_error)

            # take the next step
            current_state = next_state

        epsilon_values.append(epsilon)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

# =============================================================================
# 2. epsilon_greedy strategy func
# =============================================================================
def epsilon_greedy(state_index, Q_hat, actions, epsilon):
    """
    epsilon_greedy strategy to pick an action.

    Args:
        state_index : int
            Index of the current state.
        Q_hat : 2D numpy array
            Q values of all s,a pairs
        actions : set
            Set of actions.
        epsilon : float
            A float number to determine the exploration or explotation.
    Returns:
        action : str
            The chosen action.
    """
    if np.random.rand() < epsilon:
        # explore: random action
        return random.choice(actions)
    else:
        # exploit: choose best action
        a_idx = np.argmax(Q_hat[state_index])
        return actions[a_idx]
    
# =============================================================================
# 3. Check if epsilon decreases slowly
# =============================================================================
def implement_epsilon(epsilon_values):
    """
    Plot all epsilon values during time to check if it decreases slowly or not.

    Args:
        epsilon_values: List
            List of epsilon values.
    Returns:
        None
    """
    plt.figure(figsize=(8,4))
    plt.plot(range(len(epsilon_values)), epsilon_values, label='Epsilon')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon value')
    plt.title('Epsilon decay over episodes')
    plt.grid(True)
    plt.legend()
    plt.show()