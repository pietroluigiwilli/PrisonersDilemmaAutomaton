import numpy as np

class Environment:
    """
    This class is the environment of the iterated prisoner's dilemma (IPD) game. 
    The game and the implementation of the decision making process is explained in detail in the agent.py file.
    This class provides an interface (arena) to the game, where two agents can compete against each other.
    It takes the payoff function, a length parameter and the poisson parameter as input. The payoff function is a function that
    takes two decisions as input and returns the payoff of the two agents. The poisson parameter is used to
    add some randomness to the length of the game. The length of the game is determined by the length parameter.
    A for loop is used to iterate over the length of the game. In each iteration the agents make a decision 
    by calling: agent.decide(history_a, history_b). the decision of the agent is stored in the respective history lists,
    which are then passed to the agents in the next round. The history lists are also returned by the iterate method.
    The points computed by the payoff function are summed up and returned by the iterate method.

    ===========
    Attributes:

    payoff: function
        The payoff function is a function that takes two decisions as input and returns the payoff points of the two agents.
    
    length: int
        The length of the game. The game is played for length rounds.
    
    poisson: int
        The poisson parameter is used to add some randomness to the length of the game. 
        In this way the number of trials is unkwnown to the agents. 
    
    ========
    Methods:

    iterate(actor_a, actor_b):
        This method takes two instances of the Agent class as input and returns the total points of the two agents,
        the history lists of the two agents and the length of the game.
    """

    def __init__(self, payoff, length, poisson: int=None):
        self.payoff = payoff
        self.poisson = poisson
        self.length = length

    def iterate(self, actor_a, actor_b):
        history_a = []
        history_b = []
        if self.poisson != None:
            length = abs(np.random.randint(self.length-self.poisson, self.length+self.poisson, 1))
        else:
            length = self.length
        total_a = 0
        total_b = 0
        for i in range(length):
            decision_a = actor_a.decide(history_a, history_b)
            decision_b = actor_b.decide(history_b, history_a)
            history_a.append(decision_a)
            history_b.append(decision_b)
            
            a, b = self.payoff(decision_a, decision_b)
            total_a += a
            total_b += b
        
        return total_a, total_b, history_a, history_b