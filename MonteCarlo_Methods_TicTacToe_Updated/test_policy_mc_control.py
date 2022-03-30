
from off_monte_carlo_control_Steven import OFF_policy_monte_carlo
from off_policy_monte_carlo_control_steven import off_policy_monte_carlo

from off_policy_monte_carlo_control_steven import PolicyAndActionValueFunction
from TicTacToe import TicTacToe
from line_world import LineWorld
import numpy as np

from on_policy_first_visit_monte_carlo_control import on_policy_first_visit_monte_carlo_control

def play(pi):
    env = TicTacToe(3)
    player = True
    while not env.is_game_over():
            
        if player:
            posibilites = env.available_actions_ids()
            rand_act = np.random.choice(posibilites)
            env.play_act(rand_act,2)
            env.view()
            print("\n")
            player = not(player)
        else:
            # try:
            #     act = pi[env.state_id()]
            #     env.play_act(act,1)
            # except:
            actDict = pi[env.state_id()]

            act = list(actDict.keys())[np.argmax(list(actDict.values()))]
            # if act not in env.available_actions_ids():
            #     print(env.available_actions_ids())
            #     print(actDict)
            #     print(act)
                
            env.play_act(act, 1)
            
            env.view()
            print("\n")
            player = not(player)
    if env.other_win:
        print("Winner is Random PLayer " )
    elif env.you_win:
        print("You are the Winner" )
    else:
        print("It is a draw")
            
   

if __name__ == "__main__":
    # env = LineWorld(5,50)
    env = TicTacToe(3)

    # print(env.state_id())
    
    # env.reset_random()
    # print(env.state_id())
    # env.act_with_action_id(0,1)
    # print(env.state_id())
    result = on_policy_first_visit_monte_carlo_control(env, 0.01, 0.99999, 100000)

    # print('\n\n')
    # result = OFF_policy_monte_carlo(env, 0.1, 0.99999, 10000)
    # for state, action_values in Q.items():
    #     print(action_values)
    # print(result.pi)
    # print()
    # for k, v in result.pi.items():
    #     print(k, v)
    nbGames = 10
    for i in range(nbGames):
        print("Game: ", i)
        play(result.pi)
        print("Score: ", env.score())
        print("")