import numpy as np

from drl_sample_project_python.drl_lib.do_not_touch.contracts import SingleAgentEnv
from drl_sample_project_python.drl_lib.do_not_touch.result_structures import PolicyAndActionValueFunction


def OFF_policy_monte_carlo(
    env: SingleAgentEnv,
    epsilon: float,
    gamma : float, # on le prends proche de 1 generalement
    max_iter: int) -> PolicyAndActionValueFunction:
    
    C ={}
    Q = {}
    pi = {}
    b = {}

    for i in range(max_iter):
        env.reset()
        S = []
        A = []
        R = []
        while not env.is_game_over():
            s = env.state_id()
            S.append(s)
            available_actions =  env.available_actions_ids()
            if s not in pi:
                pi[s] = {}
                b[s] = {}
                Q[s] = {}
                C[s] = {}
                
                for a in available_actions:
                    Q[s][a] = 0.5
                    C[s][a] = 0
                    b[s][a] = 1./len(available_actions)
            #  a revoir l'emplacement de argmax
                # pi[s][a] = list(Q[s].keys())[np.argmax(list(Q[s].values()))]
                optimal_a_t = list(Q[s].keys())[np.argmax(list(Q[s].values()))]
                for a_key, q_s_a in Q[s].items():
                    
                    if a_key == optimal_a_t:
                        pi[s][a_key] = 1.0
                    else:
                        pi[s][a_key] = 0.0
       
            chosen_action = np.random.choice(
                list(b[s].keys()),
                1,
                False,
                p=list(b[s].values())
            )[0]

            A.append(chosen_action)
            old_score = env.score()
            env.act_with_action_id(chosen_action)
            # print(env.state_id())
            r = env.score() - old_score
            R.append(r)

        G = 0
        W = 1
        for t in reversed(range(len(S))):
            G = gamma * G + R[t]
            s_t = S[t]
            a_t = A[t]
            C[s_t][a_t] += W
            Q[s_t][a_t] += (W/C[s_t][a_t]) * (G -Q[s_t][a_t])
            optimal_a_t = list(Q[s_t].keys())[np.argmax(list(Q[s_t].values()))]
            pi[s_t][a_t] =optimal_a_t
            # for a_key, q_s_a in Q[s_t].items():
                    
            #         if a_key == optimal_a_t:
            #             pi[s_t][a_key] = 1.0
            #         else:
            #             pi[s_t][a_key] = 0.0
            

            if a_t != pi[s_t]:
                break
            W = W * (1./b[s_t][a_t])
    
    return PolicyAndActionValueFunction(pi, Q)