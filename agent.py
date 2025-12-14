# ==========================================================
# This file includes the agent class
# ==========================================================

# =============================================================================
# 0. Libraries
# =============================================================================
import numpy as np
import random
from itertools import product
# =============================================================================
# 1. Agent class
# =============================================================================
class agent_class:
    def __init__(self, S, A, gamma, terminal_state_idx, EPISODE_BLOCK, total_interaction, calculate_return_immediate, epsilon_decay, T_decay, lambda_value):
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
        self.T_decay = T_decay
        self.lambda_value = lambda_value
        self.list_of_returns = []

    
        # Check discount factor validity
        if not (0 <= gamma <= 1):
            raise ValueError("Discount factor γ must be between 0 and 1")
        

    def Q_learning_func(self, obj_env, discount_value, initial_state_idx , select_action_strategy, T=1):
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
        T_min = 0.05
        epsilon_values = [] # to check the condition of epsilon later
        total_interaction_manual = 0 # in all episodes

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
                    a_idx = self.boltzmann(s_idx, Q_hat, self.A, T) 

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
            total_interaction_manual += step

            # save epsilon
            epsilon_values.append(epsilon)
        
            # Decreasing epsilon for epsilon greedy
            epsilon = max(epsilon_min, epsilon * self.epsilon_decay)

            # Decreasing temperture
            T = max(T_min, T * self.T_decay)

        # end of a block of episodes
        
        if self.total_interaction:
           total_interaction = self.total_interaction
        else:
            total_interaction = total_interaction_manual

        return Q_hat, epsilon_values, total_interaction
    def sarsa(self):

        # Initializing
        Q_hat = np.zeros((len(self.S),len(self.A)), dtype=float)
        num_visit = np.zeros((len(self.S),len(self.A)), dtype=int)
        epsilon = 1.0           # initial epsilon
        epsilon_min = 0.01       # optional lower bound
        T_min = 0.05
        epsilon_values = [] # to check the condition of epsilon later
        total_interaction_manual = 0 # in all episodes

        # inside of a block of episodes
        for episode in range(self.EPISODE_BLOCK):
            pass


    
    def Evidence_of_eligibility_func(self, obj_env, gamma, lambda_value, initial_state_idx, select_action_strategy, T=1, num_episode=20):
        # Initializing
        Q_hat = np.zeros((len(self.S),len(self.A)), dtype=float)
        num_visit = np.zeros((len(self.S),len(self.A)), dtype=int)
        episode_counter=0
        epsilon_values = []
        total_interaction_manual = 0

        while True: # loop of episode
            e = np.zeros((len(self.S),len(self.A)), dtype=float)
            step = 0
            R = 0
            epsilon = 1.0           # initial epsilon
            epsilon_min = 0.01       # optional lower bound
            T_min = 0.05
            done = False
            
            # choose the start state
            if initial_state_idx is not None:
                s_idx = initial_state_idx
            else:
                s_idx = np.random.randint(0,len(self.S))

            # choose an action by one of the strategies for the current action
            if select_action_strategy == "epsilon_greedy":
                a_idx = self.epsilon_greedy(s_idx, Q_hat, self.A, epsilon) 
            
            elif select_action_strategy == "Boltzmann":
                a_idx = self.boltzmann(s_idx, Q_hat, self.A, T) 

            else:
                raise ValueError("No action strategy was provided.")
            
            while not done: # loop over interactions

                # observe st+1 and rt
                next_state_idx , reward = obj_env.interaction(s_idx, a_idx)

                # choose an action by one of the strategies for the next action
                if select_action_strategy == "epsilon_greedy":
                    next_a_idx = self.epsilon_greedy(next_state_idx, Q_hat, self.A, epsilon) 
                
                elif select_action_strategy == "Boltzmann":
                    next_a_idx = self.boltzmann(next_state_idx, Q_hat, self.A, T) 

                else:
                    raise ValueError("No action strategy was provided.")
                
                # find star action
                best_action_index = np.argmax(Q_hat[next_state_idx,:])

                # calculate the TD error
                TD_error = reward + (gamma * Q_hat[next_state_idx,best_action_index]) - Q_hat[s_idx, a_idx]

                # increase e
                e[s_idx,a_idx] += + 1

                # calculate learning step
                if obj_env.determinist == False: # stochastic
                    learning_step = 1/(num_visit[s_idx,a_idx] + 1)
                else: # deterministic
                    learning_step = 1

                # Third loop
                for s, a in product(self.S, self.A):
                    
                    Q_hat[s, a] += learning_step * TD_error * e[s, a]
                    e[s, a] = e[s, a] * gamma * lambda_value
                
                # Calculate reward immediate
                if self.calculate_return_immediate:
                    R += self.calculate_return_immediate_func(self.gamma, step, reward)

                # check the stop condition
                done = self.stop_condition(obj_env.gamma, step, s_idx, self.total_interaction)

                step += 1
                num_visit[s_idx,a_idx] += 1
            
            # end of an episode
            self.list_of_returns.append(R)
            
            # save the total interaction
            total_interaction_manual += step

            # save epsilon
            epsilon_values.append(epsilon)
        
            # Decreasing epsilon for epsilon greedy
            epsilon = max(epsilon_min, epsilon * self.epsilon_decay)

            # Decreasing temperture
            T = max(T_min, T * self.T_decay)
            #print(episode_counter)
            episode_counter += 1
            
            if episode_counter >= num_episode:
                break
            
        if self.total_interaction:
           total_interaction = self.total_interaction
        else:
            total_interaction = total_interaction_manual

        return Q_hat, epsilon_values, total_interaction
    
    def epsilon_greedy(self, s_idx, Q_hat, A, epsilon):
        if np.random.rand() < epsilon:
            # explore: random action
            return np.random.randint(0, len(A))
        else:
            # exploit: choose best action
            a = np.argmax(Q_hat[s_idx])
            return a
        
    def boltzmann(self, s_idx, Q_hat, A, T):
            
        """
        Boltzmann (softmax) action selection.

        Args:
            s_idx : int
                Current state index
            Q_hat : np.ndarray
                Q-table of shape (num_states, num_actions)
            A : np.ndarray or list
                Action space
            T : float
                Temperature (T > 0)

        Returns:
            a_idx : int
                Selected action index
        """
        if T <= 0:
            raise ValueError("Temperature T must be > 0")

        # Extract Q-values for the current state
        q_values = Q_hat[s_idx]

        # Numerical stability trick: subtract max
        q_values_stable = q_values - np.max(q_values)

        # Compute softmax probabilities
        exp_q = np.exp(q_values_stable / T)
        probs = exp_q / np.sum(exp_q)

        # Sample action according to the probabilities
        a_idx = np.random.choice(len(A), p=probs)

        return a_idx
    
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
            Q_hat, epsilon_values, total_interaction = self.Q_learning_func(env_obj, self.gamma, initial_state_idx, action_strategy, T=10)
            return Q_hat, epsilon_values, total_interaction, self.list_of_returns
        
        elif RL_method == "SARSA":
            self.sarsa()

        elif RL_method == "Evidence_of_eligibility":
            Q_hat, epsilon_values, total_interaction = self.Evidence_of_eligibility_func(env_obj, self.gamma, self.lambda_value, initial_state_idx, action_strategy, T=10, num_episode=20)
            return Q_hat, epsilon_values, total_interaction, self.list_of_returns
        else:
            raise ValueError("No RL method was provided.")