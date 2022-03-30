from line_world import LineWorld
from TicTacToe import TicTacToe
from sarsa_stevenImplem import sarsa

if __name__ == "__main__":
    # env = LineWorld(10,50)
    env = TicTacToe(3)
    print(env.__str__())
    # result = sarsa(env,0.1, 0.1, 0.9, 1000)
    # print(result.pi)