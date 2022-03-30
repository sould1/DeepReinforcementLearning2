from collections import defaultdict
from dataclasses import  dataclass
from typing import Dict
import numpy as np
from single_agent_env import SingleAgentEnv


@dataclass
class PolicyAndActionValueFunction:
    pi: Dict[int, Dict[int, float]]
    q: Dict[int, Dict[int, float]]

def on_policy_first_visit_monte_carlo_control(
        env: SingleAgentEnv,
        epsilon: float,
        gamma : float, # on le prends proche de 1 generalement
        max_iter: int) -> PolicyAndActionValueFunction:
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
            if s not in pi:
                pi[s] = {}
                q[s] = {}
                returns[s] ={}
                for a in available_actions:
                    pi[s][a] = 1./len(available_actions)
                    q[s][a] = 0.9 # peut etre pas la meilleure idee, voire ce que ca fait avec une val haute
                    returns[s][a] = [] # regarder le bouquin pour ne pas avoir une liste qui grandis a l'infini en faisant
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
                optimal_a_t = list(q[s_t].keys())[np.argmax(list(q[s_t].values()))]
                available_actions_t_count = len(q[s_t])
                for a_key, q_s_a in q[s_t].items():
                    
                    if a_key == optimal_a_t:
                        pi[s_t][a_key] = 1-epsilon + (epsilon / available_actions_t_count)
                    else:
                        pi[s_t][a_key] = epsilon/ available_actions_t_count
    # print(q)
    return PolicyAndActionValueFunction(pi, q)