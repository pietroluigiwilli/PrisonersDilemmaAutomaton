import numpy as np

class Agent:
    """
    An agent is a player in the iterated prisoner's dilemma (IPD) game.
    The IDP game is a two-player game where each player can choose to cooperate or defect.
    The decision is made simultaneously and the players are not allowed to communicate.
    The players are allowed to remember the history of the game. The game is iterated.
    Each player's (agent) objective is to maximize their own score. the score is calculated
    based on a payoff function. The payoff function usually has the following form:

        P = a if both cooperate
        P = b if both defect
        P = (c, d) if one cooperates (c) and the other defects (d), where d>c.
        Conventionally: d > a > b > c.

    The agent's decision is based on the history of the game. For example, the agent start
    by cooperating and choose to cooperate if the other agent cooperated in the previous round,
    else the agent defects. This is known as a Tit-for-Tat (tit4tat) strategy.
    The decision algorithm of the agent can be represented by a binary number with leading
    digits. The binary number must have an odd number of digits (b). This number is then
    converted to a single bit and a 2 dimensional matrix of shape (2, (b-1)/2).
        
        For example the number 8 can be converted to the 7 digit (b=7) binary number: 0001000.
        The method get_id_matrix returns the id (single bit) and the id_matrix:
        id = 0                          --> this is the first digit: [0]001000
        id_matrix = [[0, 0, 1],         --> this is the rest of the digits: 0[001000]
                     [0, 0, 0]]
    
    The id defines the first decision of the player. 0 is cooperation and 1 is defection.
    This because the actual decision is returned as such: (-1)**id. So 0 returns 1, which
    is cooperation and 1 returns -1 which is defection.
    the id_matrix defines the decision that the agent takes according to the decision that
    the other agent and itself took in the previous round. The decision is calculated as follows:

        1.  the histories of the other agent and itself are stacked in a matrix of shape (2, b-1).
            this is done in such a way that the first row is the history of the other agent.
        
        2.  the id_matrix is multiplied with the history matrix. This is done by multiplying
            the first row of the history matrix with the first row of the id_matrix and the
            second row of the history matrix with the second row of the id_matrix.
            If the history is shorter than the id_matrix, the id_matrix is sliced to the
            length of the history from the low index side (the left side is removed). 
            This is done so that the decisions are based on the most recent history.

        3.  the multiplication result is summed.

        4.  the result is passed through the heaviside function. This function returns 1 if
            the result is positive and 0 if the result is negative. 1 is returned if the
            result is 0. The result is then multiplied by 2 and 1 is subtracted.
            In this way the result is converted to -1 or 1. So if the sum is positive or 0, 
            1 is returned and the agent cooperates and if the sum is negative, 
            -1 is returned and the agent defects.
        
        5.  The decisions of the agents are stored in their respective history lists. 
            This is then passed to the agents in the next round.

    ===========
    Attributes:
    
    id: np.ndarray (private)
        The id of the agent. This is the binary number as a np.array of shape (b,).
        Each digit is in a separate element of the array. b must be an odd number.


    start: int (private)
        The first digit of the id. This is the first decision of the agent.

    id_matrix: np.ndarray (private)
        This is the decision matrix of the agent. it is a np.array of shape (2, (b-1)/2).
        it is multiplied with the history of the game to determine the decision of the agent.

    ========
    Methods:

    get_id(): 
        returns the id of the agent, which is a private attribute.

    get_id_matrix(): 
        returns the the start decision and the id_matrix of the agent.

    heaviside(x):
        returns 1 if x is positive or 0 and -1 if x is negative.
    
    decide(history_self, history_other):
        returns the decision of the agent based on the history of the game.
        The full explanation of the decision algorithm is given above.
    """

    def __init__(self, id):
        if len(id) % 2 == 0:
            raise ValueError("The id must have an odd number of digits.")
        self.__id = id
        self.__start, self.__id_matrix = self.get_id_matrix()
        self.__dim2 = self.__id_matrix.shape[1] 
        
    def get_id(self):
        return self.__id
    
    def get_id_matrix(self):
        return self.__id[0], self.__id[1:].reshape((2, (len(self.__id)-1)//2))
    
    @staticmethod
    def heaviside(x):
        return np.heaviside(x, 1)*2 - 1
    
    def decide(self, history_self, history_other):
        if len(history_other) == 0:
            return (-1)**self.__start 
        else:
            length = min(len(history_self), self.__dim2) 
            history = np.stack([history_other[-length:], history_self[-length:]])
            matrix = self.__id_matrix[:, -length:]
            decision = self.heaviside(sum(history*matrix))[0]
        return decision
    
    def __str__(self):
        return f"Agent({self.__id})"