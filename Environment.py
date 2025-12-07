# ==========================================================
# This file includes the environment class
# ==========================================================

# =============================================================================
# 0. Libraries
# =============================================================================
import numpy as np
import pulp
import matplotlib.pyplot as plt
import random
# =============================================================================
# 1. Environment clas
# =============================================================================
class MDP:
    def __init__(self, P, R, gamma, N, M, stop_condition=None):
        """
        Initialize a Markov Decision Problem (MDP)

        Args:
            P : np.ndarray
                Transition function with shape (N, M, N)
            R : np.ndarray
                Reward function with shape (N, M, N)
            gamma : float
                Discount factor in [0, 1]
            N : int
                Number of states
            M : int
                Number of actions
        Returns:
            mdp object
        """
        self.P = P          # Transition probabilities P[s, a, s']
        self.R = R          # Reward function R[s, a, s']
        self.gamma = gamma  # Discount factor
        self.N = N          # Number of states
        self.M = M          # Number of actions
        self.S = self._create_states()
        self.A = self._create_actions()
        self.stop_condition = stop_condition

        # Check discount factor validity
        if not (0 <= gamma <= 1):
            raise ValueError("Discount factor γ must be between 0 and 1")

    # ---------- Create states -------------
    def _create_states(self):
        """Create the set of states S = {s0, s1, ..., s(N-1)}"""
        return [f"s{i}" for i in range(self.N)]

    # ---------- Create actions -------------
    def _create_actions(self):
        """Create the set of actions A = {a0, a1, ..., a(M-1)}"""
        return [f"a{j}" for j in range(self.M)]
    
    def step(self, s, a):
        """
        response of env to agent taking action 'a' in a state 's' "

        Args:
            s : str
                The current state like s1
            a : str
                Current action which is taken by agent like a1
            
        Returns:
            r : the immediate reward.
            s t+1 : the next state.
        """
        s_idx = self.S.index(s)
        a_idx = self.A.index(a)

        # deterministically get next state
        next_s_idx = np.random.choice(self.N, p=self.P[s_idx, a_idx])
        next_state = self.S[next_s_idx]

        # reward for that transition
        reward = self.R[s_idx, a_idx, next_s_idx]

        done = False
        
        if self.stop_condition is not None:
            done = self.stop_condition(next_state)

        return next_state, reward, done
    
# =============================================================================
# 2. File reading Function
# =============================================================================
def read_mdp_file(path="data.txt", N=1, M=1):
    """
    Reads an MDP file with transition and reward data.
    
    Args:
        path (str): path to text file.
        N (int): number of states.
        M (int): number of actions.
    
    Returns:
        P (np.ndarray): transition matrix of shape (N, M, N)
        R (np.ndarray): reward matrix of shape (N, M, N)
    """
    with open(path, 'r') as f:
        content = f.read().strip().split('\n\n')  # split at blank line

    # Transition function
    P_lines = content[0].strip().split('\n')
    P_values = [list(map(float, line.split())) for line in P_lines]
    P = np.array(P_values).reshape(M, N, N)  # shape (M, N, N)
    P = np.transpose(P, (1, 0, 2))  # shape (N, M, N) => (state, action, next_state)

    # Normalize rows so that sum = 1
    for s in range(N):
        for a in range(M):
            P[s, a, :] = normalize_transition_matrix(P[s, a, :].reshape(1, -1))[0]

    # Reward function
    R_lines = content[1].strip().split('\n')
    R_values = [list(map(float, line.split())) for line in R_lines]
    R = np.array(R_values).reshape(M, N, N)
    R = np.transpose(R, (1, 0, 2))  # shape (N, M, N)

    return P, R

# check if all probabilities' sum is equal to 1
def normalize_transition_matrix(P, tol=1e-8):
    """
    Ensure each row of P sums to 1 by adjusting the last element.
    
    Args:
        P: Transition function.
        tol: tolerance.
    
    Returns:
        P : normalized P
    """
    for row in P:
        total = np.sum(row)
        if abs(total - 1.0) > tol:
            row[-1] = max(0.0, min(1.0, 1.0 - np.sum(row[:-1])))
    return P