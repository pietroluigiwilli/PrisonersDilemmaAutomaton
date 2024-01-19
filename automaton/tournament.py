import numpy as np
from tqdm import tqdm
from agent import Agent
from environment import Environment

class Tournament():
    """
    This class is the tournament of the iterated prisoner's dilemma (IPD) game. More information about the game
    and the implementation of the decision making process can be found in the agent.py and environment.py files.
    This class generates agents with different ids and lets them compete against each other. The ids are generated
    by counting up from 0 to the number of competitors. The decimal count is then converted to a binary number, with
    leading digits. the number of leading digits is determined by the number of competitors. specifically, by taking
    the ceiling of the log2 of the number of competitors. In order to fill the search space (number of 1's in the binary
    number being equal to the number of digits in the binary number), the number of competitors should be an odd power of 2.
    The binary number is converted to a np.array of shape (b,), where b is the number elements in the array. 
    All the agents of the search space are then generated and compete against each other. The results are stored in a np.array
    of shape (n**2, 6), where n is the number of competitors. In every competition the agents are initialized with their 
    respective ids.

    ===========
    Attributes:

    competitors: int
        The number of competitors. This is the number of agents that are generated and compete against each other. 
        For most efficient exploration of the search space, this should be an odd power of 2.

    length: int
        The length of the game. The game is played for length rounds.
    
    payoff: function
        The payoff function is a function that takes two decisions as input and returns the payoff points of the two agents.
    
    poisson: int
        The poisson parameter is used to add some randomness to the length of the game. 
        In this way the number of trials is unkwnown to the agents.
    
    ========
    Methods:

    convert_to_id(n):
        This method takes a decimal number as input and converts it to a binary number. The binary number is returned as a np.array
        of shape (b,), where b is the number of elements in the array. The binary number is also returned as a string.

    compete():
        This method generates all the agents of the search space and lets them compete against each other. The results are stored
        in a np.array of shape (n**2, 6), where n is the number of competitors. The results are returned by the method.    
    """

    def __init__(self, competitors, length, payoff, poisson: bool=None):
        self.competitors = competitors
        self.length = length
        self.environment = Environment(payoff, length, poisson=None)

    def convert_to_id(self, n):
        leading = np.log2(self.competitors).__ceil__()
        if leading%2 == 0:
            leading += 1
            print("Warning: It is best if the number of competitors be an odd power of 2 (for example 2**5, 2**7 or 2**17). Incomplete exploration of the search space may otherwise occur.")
        binary = format(n, f"0{leading}b")
        return np.array(list(binary)).astype(int), binary
        
    def compete(self):
        scores = []
        for n_a in tqdm(range(self.competitors)):
            id_a, string_a = self.convert_to_id(n_a)
            a = Agent(id_a)
            for n_b in range(self.competitors):
                id_b, string_b = self.convert_to_id(n_b) 
                b = Agent(id_b)
                score_a, score_b, _, _ = self.environment.iterate(a, b)
                scores.append([n_a, n_b, string_a, string_b, score_a, score_b])
        scores = np.array(scores)
        return scores 
    