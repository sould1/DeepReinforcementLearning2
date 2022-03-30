import dataclasses
from typing import Dict
import numpy as np
from single_agent_env import SingleAgentEnv

@dataclasses
class PolicyAndActionValueFunction:
    pi: Dict[int, Dict[int, float]]
    q: Dict[int, Dict[int, float]]
def monte_carlo_es(
    env: SingleAgentEnv,
    epsilon: float,
    gamma : float, # on le prends proche de 1 generalement
    max_iter: int
)-> PolicyAndActionValueFunction:
    pi = {}
    q = {}
    returns = {}

    
    for i in range(max_iter):
        env.reset()
        S = []
        A = []
        R = []
        while not env.is_game_over():
            s = env.state_id()
            S.append(s)
            available_actions = env.available_actions_ids()
            print(available_actions)
            if s not in pi:
                pi[s] = {}
                q[s] = {}
                returns[s] ={}
                for a in available_actions:
                    pi[s][a] = 1.0/len(available_actions)
                    q[s][a] = 0.3 # peut etre pas la meilleure idee, voire ce que ca fait avec une val haute
                    returns[s] = [] # regarder le bouquin pour ne pas avoir une liste qui grandis a l'infini en faisant
                                    # une moyenne

            chosen_action = np.random.choice(
                list(pi[s].keys()),
                1,
                False,
                p=list(pi[s].values())
            )[0]
            A.append(chosen_action)
            old_score = env.score()
            env.act_with_action_id(chosen_action)
            r = env.score() - old_score
            R.append(r)
        G = 0
        for t in reversed(range(len(S))):
            G = gamma * G + R[t]
            s_t = S[t]
            a_t = A[t]
            found = False
            for prev_s, prev_a in zip(S[:t], A[:t]):
                if s_t == prev_s and a_t == prev_a:
                    found = True
                    break
                if found:
                    continue
                returns[s_t][a_t].append(G)
                q[s_t][a_t] = np.mean(returns[s_t][a_t])
                pi[s_t] = list(q[s_t].keys())[np.argmax(list(q[s_t].values()))]
    return PolicyAndActionValueFunction(pi, q)