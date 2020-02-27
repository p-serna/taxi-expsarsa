import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, alpha = 0.3, gamma=1.0, epsilon = 1e-4,
               epsilon_min = 1e-4, epsilon_decay = 1.0):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
    def get_probs_epsgreedy(self, state):
        ''' obtain probabilities from Q using epsilon-greedy policy
        '''
        
        # We divide epsilon by nA
        policy_s = np.ones(self.nA)*self.epsilon/self.nA
        # Then the rest of the probability goes to the 'best' action
        policy_s[np.argmax(self.Q[state])] = 1 - self.epsilon + self.epsilon/self.nA

        return policy_s        
        
    def get_expectedQ(self, state):
        ''' Provides the expected value of Q[state] with probabilities from epsgreedy
        '''
        policy_s = self.get_probs_epsgreedy(state)
        return (policy_s*Q[state]).sum()
    
    def select_action(self, state):
        """ Given the state, select an action using epsilon-greedy policy.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        
        action = np.random.choice(np.arange(self.nA),
                                 p = self.get_probs_epsgreedy(state),
                                 )
        
        return action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        # self.Q[state][action] += 1
        if done:
            # Q[next_state] = 0, it is finished!
            self.Q[state][action] += self.alpha*(reward - self.Q[state][action])
        else:
            # We get the expected value from the eps-greedy policy
            policy_s = self.get_probs_epsgreedy(next_state)
            expectedQ = (policy_s*self.Q[next_state]).sum()
            # Update Q
            self.Q[state][action] += self.alpha*(reward+self.gamma*expectedQ
                                            -self.Q[state][action])
            # We update epsilon
            self.epsilon = np.max([self.epsilon*self.epsilon_decay,self.epsilon_min])