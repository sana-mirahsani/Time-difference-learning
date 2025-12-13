# ==========================================================
# This file includes solving taxi driver by Q-learning
# ==========================================================

# =============================================================================
# 0. Libraries
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from Q_learning import Q_learning_func, epsilon_greedy, implement_epsilon, derive_optimal_policy
from Environment import MDP, read_mdp_file
from agent import agent_class
# =============================================================================
# 1. Taxi driver
# =============================================================================
# create stop condition 
def my_stop(gamma, t, current_state):
    return gamma**t <= 1e-6

def create_taxi_driver_problem():
    print('====== Taxi driver ======')
    # read data
    print("Read data...")
    transition_func, reward_function = read_mdp_file("data/taxi_driver_data.txt", 3, 3)
    print("Done!")

    # create env object
    print("Create MDP object...")
    states = np.array([0,1,2])
    actions = np.array([0,1,2])
    gamma = 0.9
    taxi_obj = MDP(transition_func, reward_function, gamma, states, actions)
    print("Done!")

    # implement Q_Learning
    print("Create Q_table...")
    optimal_policy = [1,2,1]
    Q = [
        [ 88.730 , 98.800 , 84.670 ],
        [ 73.985 ,  0.000 , 76.000 ],
        [ 82.955 , 85.200 , 89.8425]
        ]
    obj_agent = agent_class(states, actions, gamma= 0.9)
    Q_hat, epsilon_values, num_step = obj_agent.Q_learning_func(taxi_obj, 0.9, None, "epsilon_greedy", my_stop, Q, tolerance=30)
    
    implement_epsilon(epsilon_values)
    print("Done!")

    #derive_optimal_policy(Q_hat,states,actions)

create_taxi_driver_problem()

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

def derive_optimal_policy(Q_hat,states,actions):
    """
    Drive the optimal policy from the Q_hat table after all iterations.

    Args:
        Q_hat: 2D numpy array
                A Table of Q values for all s,a pairs.
        states: List
                A list of states.
        actions :List
                A list of actions.
    Returns:
        Optimal_policy : A list
    """
    optimal_policy = []
    states = list(states)
    actions = list(actions)

    for s in states:
        s_idx = states.index(s) # find the index

        best_action_index = np.argmax(Q_hat[s_idx,:])
        optimal_policy.append(best_action_index)

        print(f"For state : {s}, best action is : {actions[best_action_index]}")

    print("=========== Optimal policy ===========")
    print(optimal_policy)

    return optimal_policy