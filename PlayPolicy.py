import pickle
import numpy as np
from drl_sample_project_python.drl_lib.to_do.TicTacToe import TicTacToe

def play(pi, nameAlgo):
    print("Algo: ", nameAlgo )
    env = TicTacToe(3)
    randPlayer = False
    # env.reset_random()
    # env.view()
    while not env.is_game_over():
        actDict = pi[env.state_id()]
        act = list(actDict.keys())[np.argmax(list(actDict.values()))]
        env.act_with_action_id(act)
        env.view()
    if env.other_win:
        print("Winner is Random PLayer ")
    elif env.you_win:
        print("You are the Winner")
    else:
        print("It is a draw")

if __name__ == "__main__":
    loadSarsa = pickle.load(open("sarsa_on_TTT.sav", "rb"))
    play(loadSarsa.pi, "Sarsa")
    loadQ = pickle.load(open("qlearning1.sav", "rb"))
    play(loadQ.q, "Q-Learning")
    loadExpectedSarsa = pickle.load(open("expected-sarsa.sav", "rb"))
    play(loadExpectedSarsa.q, "Expected-Sarsa")