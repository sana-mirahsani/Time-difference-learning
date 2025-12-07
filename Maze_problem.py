# ==========================================================
# This file includes solving Maze by Q-Learning
# ==========================================================

# =============================================================================
# 0. Libraries
# =============================================================================
import numpy as np
import pulp
import matplotlib.pyplot as plt
from Q_learning import Q_learning_func, epsilon_greedy, implement_epsilon, derive_optimal_policy
from Environment import MDP, read_mdp_file

# =============================================================================
# 1. Maze
# =============================================================================

def Maze_func():
    print('====== Taxi driver ======')
    # read data
    print("Read data...")
    transition_func, reward_function = read_mdp_file("data/labyrinthes.txt", 24, 4)
    print("Done!")

    # create env object
    print("Create MDP object...")
    taxi_obj = MDP(transition_func, reward_function, 0.95, 24, 4)
    print("Done!")