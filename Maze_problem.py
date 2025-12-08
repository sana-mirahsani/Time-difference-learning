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
    maze_obj = MDP(transition_func, reward_function, 0.95, 24, 4)
    print("Done!")
    maze_obj.A = ["left", "right", "up", "down"]
    maze_obj.S = {
    0: {"coord": (0,0), "type": "wall"},
    1: {"coord": (0,1), "type": "wall"},
    2: {"coord": (0,2), "type": "wall"},
    3: {"coord": (0,3), "type": "wall"},
    4: {"coord": (0,4), "type": "wall"},
    5: {"coord": (0,5), "type": "wall"},
    6: {"coord": (0,6), "type": "wall"},

    7: {"coord": (1,6), "type": "wall"},
    8: {"coord": (2,6), "type": "wall"},
    9: {"coord": (3,6), "type": "wall"},
    10: {"coord": (4,6), "type": "wall"},
    11: {"coord": (5,6), "type": "wall"},
    12: {"coord": (6,6), "type": "wall"},
    13: {"coord": (7,6), "type": "wall"},
    14: {"coord": (8,6), "type": "wall"},

    15: {"coord": (8,5), "type": "wall"},
    16: {"coord": (8,4), "type": "wall"},
    17: {"coord": (8,3), "type": "wall"},
    18: {"coord": (8,2), "type": "wall"},
    19: {"coord": (8,1), "type": "wall"},
    20: {"coord": (8,0), "type": "wall"},

    21: {"coord": (7,0), "type": "wall"},
    22: {"coord": (6,0), "type": "wall"},
    23: {"coord": (5,0), "type": "wall"},
    24: {"coord": (4,0), "type": "wall"},
    25: {"coord": (3,0), "type": "wall"},
    26: {"coord": (2,0), "type": "wall"},
    27: {"coord": (1,0), "type": "wall"},

    28: {"coord": (1,1), "type": "free"},
    29: {"coord": (1,2), "type": "free"},
    30: {"coord": (1,3), "type": "free"},
    31: {"coord": (1,4), "type": "wall"},
    32: {"coord": (1,5), "type": "free"},

    33: {"coord": (2,5), "type": "free"},
    34: {"coord": (3,5), "type": "free"},
    35: {"coord": (4,5), "type": "free"},
    36: {"coord": (5,5), "type": "free"},
    37: {"coord": (6,5), "type": "free"},
    38: {"coord": (7,5), "type": "green"},

    39: {"coord": (7,4), "type": "free"},
    40: {"coord": (7,3), "type": "free"},
    41: {"coord": (7,2), "type": "free"},
    42: {"coord": (7,1), "type": "free"},

    43: {"coord": (6,1), "type": "free"},
    44: {"coord": (5,1), "type": "free"},
    45: {"coord": (4,1), "type": "free"},
    46: {"coord": (3,1), "type": "free"},
    47: {"coord": (2,1), "type": "free"},

    48: {"coord": (2,2), "type": "wall"},
    49: {"coord": (2,3), "type": "free"},
    50: {"coord": (2,4), "type": "free"},

    51: {"coord": (3,4), "type": "wall"},
    52: {"coord": (4,4), "type": "free"},
    53: {"coord": (5,4), "type": "wall"},
    54: {"coord": (6,4), "type": "wall"},

    55: {"coord": (6,3), "type": "free"},
    56: {"coord": (6,2), "type": "wall"},

    57: {"coord": (5,2), "type": "wall"},
    58: {"coord": (4,2), "type": "orange"},
    59: {"coord": (3,2), "type": "wall"},

    60: {"coord": (3,3), "type": "wall"},
    61: {"coord": (4,3), "type": "wall"},
    
    }

    print(maze_obj.A)
    print(maze_obj.S)

Maze_func()