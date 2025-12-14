# ==========================================================
# This file includes the agent class
# ==========================================================

# =============================================================================
# 0. Libraries
# =============================================================================
import numpy as np
import random

# =============================================================================
# 1. Agent class
# =============================================================================
class agent_class:
    def __init__(self, S, A, gamma, terminal_state_idx, EPISODE_BLOCK, total_interaction, calculate_return_immediate, epsilon_decay):
        """
        Initialize an agent.

        Args:
            S : np.ndarray
                Array of states
            A : np.ndarray
                Array of actions
            gamma : float
                Discount factor in [0, 1]
        
        Returns:
            agent object
        """
        self.S = S  # states
        self.A = A  # actions
        self.gamma = gamma  # Discount factor
        self.terminal_state_idx = terminal_state_idx
        self.EPISODE_BLOCK = EPISODE_BLOCK
        self.total_interaction = total_interaction
        self.calculate_return_immediate = calculate_return_immediate
        self.epsilon_decay = epsilon_decay
        self.list_of_returns = []
    
        # Check discount factor validity
        if not (0 <= gamma <= 1):
            raise ValueError("Discount factor γ must be between 0 and 1")
        

    def Q_learning_func(self, obj_env, discount_value, initial_state_idx , select_action_strategy, T=1000):
        """
        Q_learning algorithm.

        Args:
            obj_env : object
                An object from environment.
            discount_value : float
                A float number.
            select_action_strategy : function
                A function to choose the action.
            max_episodes : int
                Number of max iteration.

        Returns:
            Q_table : numpy array 2D
                The Q value for all states and actions pairs.
            epsilon_values: A list
                List of all epsilons.
        """
        
        # Initializing
        Q_hat = np.zeros((len(self.S),len(self.A)), dtype=float)
        num_visit = np.zeros((len(self.S),len(self.A)), dtype=int)
        epsilon = 1.0           # initial epsilon
        epsilon_min = 0.01       # optional lower bound
        epsilon_values = [] # to check the condition of epsilon later
        total_interaction = 0 # in all episodes

        # inside of a block of episodes
        for episode in range(self.EPISODE_BLOCK):
            
            # choose the start state
            if initial_state_idx is not None:
                s_idx = initial_state_idx
            else:
                s_idx = np.random.randint(0,len(self.S))

            done = False
            step = 0 # in one episode
            R = 0
        
            # Inside an episode
            while not done:
                
                # choose an action by one of the strategies
                if select_action_strategy == "epsilon_greedy":
                    a_idx = self.epsilon_greedy(s_idx, Q_hat, self.A, epsilon) 
                
                elif select_action_strategy == "Boltzmann":
                    a_idx = self.boltzmann(self.s, Q_hat, self.A, epsilon, T) 

                else:
                    raise ValueError("No action strategy was provided.")
                
                # observe st+1 and rt
                next_state_idx , reward = obj_env.interaction(s_idx, a_idx) 
                
                # calculate the TD error
                TD_error = reward + (discount_value * (np.max(Q_hat[next_state_idx,:]))) - Q_hat[s_idx, a_idx]
                
                # calculate learning step
                if obj_env.determinist == False: # stochastic
                    learning_step = 1/(num_visit[s_idx,a_idx] + 1)
                else: # deterministic
                    learning_step = 1

                # Calculate reward immediate
                if self.calculate_return_immediate:
                    R += self.calculate_return_immediate_func(self.gamma, step, reward)

                # update Q_hat
                Q_hat[s_idx, a_idx] = Q_hat[s_idx, a_idx] + learning_step * (TD_error)

                # update number of visited
                num_visit[s_idx,a_idx] += 1

                # check the stop condition
                done = self.stop_condition(obj_env.gamma, step, s_idx, self.total_interaction)

                # take the next step
                s_idx = next_state_idx

                step += 1 # increase 

            # end of an episode
            self.list_of_returns.append(R)
            
            # save the total interaction
            total_interaction += step

            # save epsilon
            epsilon_values.append(epsilon)
        
            # Decreasing epsilon for epsilon greedy
            epsilon = max(epsilon_min, epsilon * self.epsilon_decay)

        # end of a block of episodes
                
        return Q_hat, epsilon_values, total_interaction
    
    def sarsa(self):
        pass

    def epsilon_greedy(self, s_idx, Q_hat, A, epsilon):
        if np.random.rand() < epsilon:
            # explore: random action
            return np.random.randint(0, len(A))
        else:
            # exploit: choose best action
            a = np.argmax(Q_hat[s_idx])
            return a
        
    def boltzmann(self, s, Q_hat, A, epsilon, T):
        pass


    def Evidence_of_eligibility_func(self):
        pass
    
    def stop_condition(self, gamma, step, current_state_idx, total_interaction):
        
        if total_interaction != None: # if there is a number of interactions
            if step == total_interaction:
                return True
            
        if self.terminal_state_idx == None: # there is No terminal state
            max_steps = 5000
            return gamma**step <= 1e-6 or step >= max_steps
        else:
            if current_state_idx == self.terminal_state_idx:
                
                return True
    
    def calculate_return_immediate_func(self, gamma, step, reward):
        return (gamma ** step) * reward
    
    def trainning(self, RL_method, action_strategy, env_obj, initial_state_idx):

        if RL_method == "Q_Learning":
            Q_hat, epsilon_values, total_interaction = self.Q_learning_func(env_obj, self.gamma, initial_state_idx, action_strategy)
            return Q_hat, epsilon_values, total_interaction, self.list_of_returns
        
        elif RL_method == "SARSA":
            self.sarsa()

        else:
            raise ValueError("No RL method was provided.")