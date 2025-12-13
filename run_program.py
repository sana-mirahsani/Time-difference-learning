import numpy as np 
from Environment import environment_class
from agent import agent_class
import matplotlib.pyplot as plt
# =============================================================================
# 2. Main function
# =============================================================================
class solve_mdp():
    def __init__(self, file_name, problem_name, 
                 S, A, gamma, isdeterministic, 
                 RL_method, action_strategy, Q_optimal_policy, initial_state_idx, tolerance, terminal_state_idx, EPISODE_BLOCK):
        
        self.file_name=file_name
        self.problem_name=problem_name

        self.states=S
        self.actions=A
        self.gamma=gamma
        self.isdeterministic=isdeterministic

        self.RL_method=RL_method
        self.action_strategy=action_strategy
        self.Q_optimal_policy=Q_optimal_policy
        self.initial_state_idx=initial_state_idx 
        self.tolerance=tolerance 
        self.terminal_state_idx = terminal_state_idx
        self.EPISODE_BLOCK = EPISODE_BLOCK

    def read_mdp_file(self, path=None, N=1, M=1):
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
        if path:
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
                    P[s, a, :] = self.normalize_transition_matrix(P[s, a, :].reshape(1, -1))[0]

            # Reward function
            R_lines = content[1].strip().split('\n')
            R_values = [list(map(float, line.split())) for line in R_lines]
            R = np.array(R_values).reshape(M, N, N)
            R = np.transpose(R, (1, 0, 2))  # shape (N, M, N)

            return P, R
        else:
            raise ValueError("no file data was provide.")
        
    # check if all probabilities' sum is equal to 1
    def normalize_transition_matrix(self, P, tol=1e-8):
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
    
    def implement_epsilon(self, epsilon_values):
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

    def derive_optimal_policy(self, Q_hat, states, actions):
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

    def main(self):
        print(f'====== {self.problem_name} ======')
        # read data
        print("Read data...")
        transition_func, reward_function = self.read_mdp_file(self.file_name, len(self.states), len(self.actions))
        print("Done!")

        # create environment
        print("Create environment...")
        env_obj = environment_class(transition_func, reward_function, self.gamma, self.states, self.actions, self.isdeterministic)
        print("Done!")

        # create agent
        print("Create agent...")
        agent_obj = agent_class(self.states, self.actions, self.gamma, self.terminal_state_idx, self.EPISODE_BLOCK)
        print("Done!")

        # start training
        print("Start training...")
        Q_hat, epsilon_values, total_interaction = agent_obj.trainning(self.RL_method, self.action_strategy, self.Q_optimal_policy, env_obj, self.initial_state_idx, self.tolerance)
        print("training Done!")
        
        # implement epsilon trend
        self.implement_epsilon(epsilon_values)
        
        # Drive optimal policy
        self.derive_optimal_policy(Q_hat,self.states, self.actions)

        print("=========== Optimal Q value ===========")
        print(Q_hat)

        print("=========== total_interaction ===========")
        print(total_interaction)


mdp_problem = solve_mdp(file_name="data/taxi_driver_data.txt", problem_name="Taxi driver", 
                        S=np.array([0,1,2]), A=np.array([0,1,2]), gamma=0.9, isdeterministic=False, 
                        RL_method="Q_Learning", action_strategy="epsilon_greedy", Q_optimal_policy=None, initial_state_idx=1, tolerance=0.1, terminal_state_idx=None, EPISODE_BLOCK=50)

mdp_problem.main()