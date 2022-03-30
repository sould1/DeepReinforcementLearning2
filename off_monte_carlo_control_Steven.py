from dataclasses import  dataclass
from collections import defaultdict
from off_policy_monte_carlo_control_steven import PolicyAndActionValueFunction

from typing import Dict
import numpy as np
from single_agent_env import SingleAgentEnv



def OFF_policy_monte_carlo(
    env: SingleAgentEnv,
    epsilon: float,
    gamma : float, # on le prends proche de 1 generalement
    max_iter: int) -> PolicyAndActionValueFunction:
    nA = len(env.available_actions_ids())
    Q = {}
    C = {}
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
                Q[s] = {}
                C[s] = {}
                b[s] = {}
                for a in available_actions:
                    b[s][a] = 1./len(available_actions)
                    Q[s][a] = 0.5
                    C[s][a] = 0
            pi[s] = list(Q[s].keys())[np.argmax(list(Q[s].values()))]
       
            chosen_action = np.random.choice(
            list(b[s].keys()),
            1,
            False,
            p=list(b[s].values())
            )[0]

            A.append(chosen_action)
            old_score = env.score()
            env.act_with_action_id(chosen_action)
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
                pi[s_t] = list(Q[s_t].keys())[np.argmax(list(Q[s_t].values()))]
                if a_t != pi[s_t]:
                    break
                W = W * (1./b[s_t][a_t])
    return PolicyAndActionValueFunction(pi, Q)