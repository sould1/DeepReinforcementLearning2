from botan_expected import expected
from sarsa_stevenImplem import sarsa
from expected_sarsa_stevenImplem import expected_sarsa
from TicTacToe import TicTacToe
from line_world import LineWorld
from q_learning import q_learning
from TicTacToe import TicTacToe
from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":
    # env = LineWorld(10,50)
    env= TicTacToe(3)
    # print(env.you_win)
    # print(env.__str__().split("|"))
    # result, arr = q_learning(env,0.99, 0.0001, 0.999999, 100000)
    # result, arr = expected_sarsa(env,0.95, 1, 0.99999, 50000)
    resutl = expected(env)
    # result, arr = sarsa(env,0.1, 0.1, 0.99999, 10000)

    # print(arr)
    # plt.plot(arr)
    # plt.show()

    # print(result.pi)
    # print(result.q)