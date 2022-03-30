import numpy as np
import matplotlib.pyplot as plt

from drl_sample_project_python.drl_lib.do_not_touch.contracts import SingleAgentEnv
from drl_sample_project_python.drl_lib.do_not_touch.result_structures import PolicyAndActionValueFunction


def q_learning(
        env: SingleAgentEnv,
        alpha : float,
        epsilon: float,
        gamma : float, # on le prends proche de 1 generalement
        max_iter: int) -> PolicyAndActionValueFunction:
    assert (epsilon > 0)
    pi = {} # learned greedy policy
    b = {} # behavioour epsilon-greedy policy
    q = {} # action value function de pi
    
    cptWin = 0
    cptLose = 0
    cptDraw = 0
    ratioWinLose =0.0
    arayPlot = []
    for i in range(max_iter):
        env.reset()
        
        while not env.is_game_over():
            s = env.state_id()
            available_actions = env.available_actions_ids()
            if s not in pi:
                pi[s] = {}
                q[s] = {}
                b[s] = {}
                for a in available_actions:
                    pi[s][a] = 1.0/len(available_actions)
                    q[s][a] = 0.0 # peut etre pas la meilleure idee, voire ce que ca fait avec une val haute
                    b[s][a] = 1.0/len(available_actions)

            available_actions_count = len(available_actions)
            optimal_a = list(q[s].keys())[np.argmax(list(q[s].values()))]

            for a_key, q_s_a in q[s].items():
                if a_key == optimal_a:
                    b[s][a_key] = 1-epsilon + epsilon / available_actions_count
                else:
                    b[s][a_key] = epsilon/ available_actions_count
            chosen_action = np.random.choice(
                list(b[s].keys()),
                1,
                False,
                p=list(b[s].values())
            )[0]
            old_score= env.score()
            env.act_with_action_id(chosen_action)

            r = env.score() - old_score
            s_p = env.state_id()
            next_available_action = env.available_actions_ids()

            if env.is_game_over():
                q[s][chosen_action] += alpha * (r + 0.0 - q[s][chosen_action])

                sumUp = env.__str__().split("|")
                if sumUp[0] =="True":
                    cptWin += 1
                elif sumUp[1] == "True":
                    cptLose += 1
                else:
                    cptDraw += 1

            else:
                if s_p not in pi:
                    pi[s_p] = {}
                    q[s_p] = {}
                    b[s_p] = {}
                    for a in next_available_action:
                        pi[s_p][a] = 1.0 / len(next_available_action)
                        q[s_p][a] = 0.0  # peut etre pas la meilleure idee, voire ce que ca fait avec une val haute
                        b[s_p][a] = 1.0 / len(next_available_action)
                q[s][chosen_action] += alpha * (r + gamma * np.max(list(q[s_p].values())) - q[s][chosen_action])
        
        # if i%100 ==0 and i !=0:
        #     epsilon *= 0.005

        if i%1000 ==0 and i!=0:
            # ratioWinLose =  1 if cptLose == cptDraw == 0 else cptWin/(cptLose+cptDraw) 
            ratioWinLose = cptWin/float(cptLose+cptDraw+cptWin)
            print("Win: ",cptWin)
            print("Lose: ", cptLose)
            print("Draw: ", cptDraw)
            print("Ration Win/Lose: ", ratioWinLose)      
            arayPlot.append(ratioWinLose)
            print("Eps: ", epsilon)

    # la strategie de jeu
    for s in q.keys():
        optimal_a = list(q[s].keys())[np.argmax(list(q[s].values()))]
        for a_key, q_s_a in q[s].items():
            if a_key == optimal_a:
                pi[s][a_key] = 1.0
            else:
                pi[s][a_key] = 0.0

    plt.plot(arayPlot)
    plt.show()
    return PolicyAndActionValueFunction(pi, q)