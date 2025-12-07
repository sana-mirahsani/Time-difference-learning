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

# =============================================================================
# 1. Taxi driver
# =============================================================================
def my_stop(gamma, t, current_state):
    return gamma**t <= 1e-6

def create_taxi_driver_problem():
    print('====== Taxi driver ======')
    # read data
    print("Read data...")
    transition_func, reward_function = read_mdp_file("data/taxi_driver_data.txt", 3, 3)
    print("Done!")

    # create stop condition 
    #stop_condition_func = lambda *args, **kwargs: pow(kwargs.get('gamma', 1), kwargs.get('t', 0)) <= 1e-6

    # create env object
    print("Create MDP object...")
    taxi_obj = MDP(transition_func, reward_function, 0.9, 3, 3)
    print("Done!")

    # implement Q_Learning
    print("Create Q_table...")
    Q_table, epsilon_values, num_steps = Q_learning_func(obj_env=taxi_obj, discount_value=0.9, select_action_strategy=epsilon_greedy, stop_condition=my_stop, max_episodes=100)
    print("Done!")

    # check if the epsilon decreases correctly
    implement_epsilon(epsilon_values)

    # derive the optimal policy
    print("Derive the optimal policy...")
    optimal_policy = derive_optimal_policy(Q_table,taxi_obj.S, taxi_obj.A)
    print("Done!")

create_taxi_driver_problem()