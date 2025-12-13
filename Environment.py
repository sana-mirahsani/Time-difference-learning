# ==========================================================
# This file includes the environment class
# ==========================================================

# =============================================================================
# 0. Libraries
# =============================================================================
import numpy as np

# =============================================================================
# 1. Environment class
# =============================================================================
class environment_class:
    def __init__(self, P, R, gamma, S, A, isdeterminist):
        """
        Initialize a Markov Decision Problem (MDP)

        Args:
            P : np.ndarray
                Transition function with shape (N, M, N)
            R : np.ndarray
                Reward function with shape (N, M, N)
            gamma : float
                Discount factor in [0, 1]
            S : np.ndarray
                Array of states
            A : np.ndarray
                Array of actions
        Returns:
            mdp object
        """
        self.P = P          # Transition probabilities P[s, a, s']
        self.R = R          # Reward function R[s, a, s']
        self.gamma = gamma  # Discount factor
        self.S = S  # states
        self.A = A  # actions
        self.determinist = isdeterminist

        # Check discount factor validity
        if not (0 <= gamma <= 1):
            raise ValueError("Discount factor γ must be between 0 and 1")
    
    def interaction(self, s, a):
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
        P = self.P[s,a]
        if np.sum(P) ==1:
            next_state = np.random.choice(len(self.S), size=1, p=P)
            return next_state[0], self.R[s,a,next_state][0]

        else:
            return s, 0