
from off_monte_carlo_control_Steven import OFF_policy_monte_carlo
from off_policy_monte_carlo_control_steven import off_policy_monte_carlo
from off_policy_monte_carlo_control_steven import PolicyAndActionValueFunction
from TicTacToe import TicTacToe
from line_world import LineWorld
import numpy as np

from on_policy_first_visit_monte_carlo_control import on_policy_first_visit_monte_carlo_control

def play(pi):
    env = TicTacToe(3)
    while not env.is_game_over():
        for player in [1,2]:
            if player ==1:
                posibilites = env.available_actions_ids()
                rand_act = np.random.choice(posibilites)
                env.act_with_action_id(rand_act)
            else:
                act = pi[env.state_id()]
                env.act_with_action_id(act)
            print(env)
            if env.is_game_over():
                print("Winner is " + str(player))
                break
            
   

if __name__ == "__main__":
    # env = LineWorld(5,50)
    env = TicTacToe(3)
    # print(env.state_id())
    
    # env.reset_random()
    # print(env.state_id())
    # env.act_with_action_id(0,1)
    # print(env.state_id())
    # result = on_policy_first_visit_monte_carlo_control(env, 0.1, 0.99999, 10000)
   
    result = OFF_policy_monte_carlo(env, 0.1, 0.99999, 10000)
    # for state, action_values in Q.items():
    #     print(action_values)
    # print(result.pi.keys())
    # print()
    # for k, v in result.pi.items():
    #     print(k, v)
    play(result.pi)