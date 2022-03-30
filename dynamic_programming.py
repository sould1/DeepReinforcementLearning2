from math import sqrt
import random

import numpy as np

from . import grid_world_mdpEnv
from ..do_not_touch.mdp_env_wrapper import Env1
from ..do_not_touch.result_structures import ValueFunction, PolicyAndValueFunction



def policy_evaluation_on_line_world() -> ValueFunction:

    ######### LINE WORLD MDP DEFINITION #########

    MAX_CELLS = 7
    S = np.arange(MAX_CELLS)
    A = np.array([0, 1])  # 0: Gauche, 1: Droite
    R = np.array([-1, 0, 1])
    p = np.zeros((len(S), len(A), len(S), len(R)))

    for i in range(1, MAX_CELLS - 2):
        p[i, 1, i + 1, 1] = 1.0

    for i in range(2, MAX_CELLS - 1):
        p[i, 0, i - 1, 1] = 1.0

    p[MAX_CELLS - 2, 1, MAX_CELLS - 1, 2] = 1.0
    p[1, 0, 0, 0] = 1.0
    pi = np.zeros((len(S), len(A)))
    # pi[:, 1] = 1.0  # Aller toujours à droite ! :)
    # pi[:, 0] = 1.0  # Aller toujours à gauche ! :)
    pi[:, :] = 0.5  # Aller 50% du temps à gauche et 50% du temps à droite ! :)
    theta = 0.0001
    V = np.zeros((len(S),))
    gamma = 1.0

    ######### POLYCY EVALUATION IMPLEMENTATION #########

    while True:
        delta = 0
        for s in S:
            v = V[s]
            V[s] = 0
            for a in A:
                for s_p in S:
                    for r_idx, r in enumerate(R):
                        V[s] += pi[s, a] * p[s, a, s_p, r_idx] * (r + gamma * V[s_p])
            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            break

    ret_value = dict(enumerate(V.flatten(), 1))
    # return V
    return ret_value

    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    # TODO
    pass


def policy_iteration_on_line_world() -> PolicyAndValueFunction:

    ######### LINE WORLD MDP DEFINITION #########
    MAX_CELLS = 7
    S = np.arange(MAX_CELLS)
    A = np.array([0, 1])  # 0: Gauche, 1: Droite
    R = np.array([-1, 0, 1])
    p = np.zeros((len(S), len(A), len(S), len(R)))

    for i in range(1, MAX_CELLS - 2):
        p[i, 1, i + 1, 1] = 1.0

    for i in range(2, MAX_CELLS - 1):
        p[i, 0, i - 1, 1] = 1.0
    p[MAX_CELLS - 2, 1, MAX_CELLS - 1, 2] = 1.0
    p[1, 0, 0, 0] = 1.0

    ######### POLYCY ITERATION IMPLEMENTATION #########

    pi = np.zeros((len(S), len(A)))
    # pi[:, 1] = 1.0  # Aller toujours à droite ! :)
    # pi[:, 0] = 1.0  # Aller toujours à gauche ! :)
    pi[:, :] = 0.5  # Aller 50% du temps à gauche et 50% du temps à droite ! :)

    # 1 : Initialization
    pi = np.zeros((len(S), len(A)))
    for s in S:
        pi[s, random.randint(0, len(A) - 1)] = 1.0

    V = np.zeros((len(S),))

    while True:
        # 2 : Policy Evaluation
        theta = 0.00001
        gamma = 0.99999

        while True:
            delta = 0
            for s in S:
                v = V[s]
                V[s] = 0
                for a in A:
                    for s_p in S:
                        for r_idx, r in enumerate(R):
                            V[s] += pi[s, a] * p[s, a, s_p, r_idx] * (r + gamma * V[s_p])
                delta = max(delta, abs(v - V[s]))

            if delta < theta:
                break

        # 3 : Policy Improvement
        policy_stable = True
        for s in S:
            old_state_policy = np.copy(pi[s, :])

            best_a = -1
            best_a_score = None

            for a in A:
                a_score = 0.0
                for s_p in S:
                    for r_idx, r in enumerate(R):
                        a_score += p[s, a, s_p, r_idx] * (r + gamma * V[s_p])
                if best_a_score is None or best_a_score < a_score:
                    best_a = a
                    best_a_score = a_score
            pi[s, :] = 0.0
            pi[s, best_a] = 1.0

            if not np.array_equal(old_state_policy, pi[s]):
                policy_stable = False
        if policy_stable:
            break

    pi_dict = dict()
    for i, value in enumerate(pi):
        pi_dict[i + 1] = dict(enumerate(value, 1))

    ret_value = dict(enumerate(V.flatten(), 1))

    return pi_dict, ret_value

    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    # TODO
    pass


def value_iteration_on_line_world() -> PolicyAndValueFunction:

    ######### LINE WORLD MDP DEFINITION #########
    MAX_CELLS = 7

    S = np.arange(MAX_CELLS)
    A = np.array([0, 1])  # 0: Gauche, 1: Droite
    R = np.array([-1, 0, 1])
    p = np.zeros((len(S), len(A), len(S), len(R)))

    for i in range(1, MAX_CELLS - 2):
        p[i, 1, i + 1, 1] = 1.0

    for i in range(2, MAX_CELLS - 1):
        p[i, 0, i - 1, 1] = 1.0

    p[MAX_CELLS - 2, 1, MAX_CELLS - 1, 2] = 1.0
    p[1, 0, 0, 0] = 1.0

    ######### VALUE ITERATION IMPLEMENTATION #########
    theta = 0.000005
    gamma = 0.99999
    condition = True

    pi = np.zeros((len(S), len(A)))
    for s in S:
        pi[s, random.randint(0, len(A) - 1)] = 1.0

    V = np.zeros((len(S),))
    for indx, v in enumerate(V):
        V[indx] = random.random()

    V[0] = 0
    V[MAX_CELLS - 1] = 0

    while condition:
        delta = 0
        for s in S:
            v = V[s]

            best_a = -1
            best_a_score = None

            for a in A:
                a_score = 0.0
                for s_p in S:
                    for r_idx, r in enumerate(R):
                        a_score += p[s, a, s_p, r_idx] * (r + gamma * V[s_p])
                if best_a_score is None or best_a_score < a_score:
                    best_a = a
                    best_a_score = a_score
                    pi[s, :] = 0.0
                    pi[s, best_a] = 1.0

            V[s] = best_a_score
            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            condition = False

    pi_dict = dict()
    for i, value in enumerate(pi):
        pi_dict[i + 1] = dict(enumerate(value, 1))


    ret_value = dict(enumerate(V.flatten(), 1))

    return pi_dict, ret_value
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    # TODO
    pass


def policy_evaluation_on_grid_world() -> ValueFunction:

    ######### GRID WORD DEFINITION #########

    env = grid_world_mdpEnv.grid_world_mdpEnvv(5)
    S = env.states()
    A = env.actions()
    R = env.rewards()

        ######### VALUE ITERATION IMPLEMENTATION #########

    pi = np.zeros((len(S), len(A)))

    # pi[:, 1] = 1.0  # Aller toujours à droite ! :)
    # pi[:, 0] = 1.0  # Aller toujours à gauche ! :)
    pi[:, :] = 0.5  # Aller 50% du temps à gauche et 50% du temps à droite ! :)

    theta = 0.0001
    V = np.zeros((len(S),))
    gamma = 0.5

    while True:
        delta = 0
        for s in S:
            v = V[s]
            V[s] = 0
            for a in A:
                for s_p in S:
                    for r_idx, r in enumerate(R):
                        # print(V[s])
                        V[s] += pi[s, a] * env.transition_probability(s, a, s_p, r_idx) * (r + gamma * V[s_p])
            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            break

    ret_value = dict(enumerate(V.flatten(), 1))

    return ret_value

    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    # TODO
    pass


def policy_iteration_on_grid_world() -> PolicyAndValueFunction:

    ###### GRID WORD DEFINITION ######

    env = grid_world_mdpEnv.grid_world_mdpEnvv(5)
    S = env.states()
    A = env.actions()
    R = env.rewards()
    # p = env.transition_proba

   ##### POLYCY ITERATION IMPLEMENTATION #####
    # 1 : Initialization
    pi = np.zeros((len(S), len(A)))
    for s in S:
        pi[s, random.randint(0, len(A) - 1)] = 1.0

    V = np.zeros((len(S),))

    while True:
        # 2 : Policy Evaluation
        theta = 0.00001
        gamma = 0.99999

        while True:
            delta = 0
            for s in S:
                v = V[s]
                V[s] = 0
                for a in A:
                    for s_p in S:
                        for r_idx, r in enumerate(R):
                            V[s] += pi[s, a] * env.transition_probability(s, a, s_p, r_idx) * (r + gamma * V[s_p])
                delta = max(delta, abs(v - V[s]))

            if delta < theta:
                break

        # 3 : Policy Improvement
        policy_stable = True
        for s in S:
            old_state_policy = np.copy(pi[s, :])

            best_a = -1
            best_a_score = None

            for a in A:
                a_score = 0.0
                for s_p in S:
                    for r_idx, r in enumerate(R):
                        a_score += env.transition_probability(s, a, s_p, r_idx) * (r + gamma * V[s_p])
                if best_a_score is None or best_a_score < a_score:
                    best_a = a
                    best_a_score = a_score
            pi[s, :] = 0.0
            pi[s, best_a] = 1.0

            if not np.array_equal(old_state_policy, pi[s]):
                policy_stable = False
        if policy_stable:
            break

    pi_dict = dict()
    for i, value in enumerate(pi):
        pi_dict[i + 1] = dict(enumerate(value, 1))

    ret_value = dict(enumerate(V.flatten(), 1))

    return pi_dict, ret_value

    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    # TODO
    pass


def value_iteration_on_grid_world() -> PolicyAndValueFunction:
    print("value iteration on grid world")
    ###### GRID WORD DEFINITION ######
    MAX_CELLS = 25
    env = grid_world_mdpEnv.grid_world_mdpEnvv(5)
    S = env.states()
    A = env.actions()
    R = env.rewards()
    p = env.transition_proba



    ###### VALUE ITERATION IMPLEMENTATION ######
    theta = 0.000005
    gamma = 0.99999
    condition = True

    pi = np.zeros((len(S), len(A)))
    for s in S:
        pi[s, random.randint(0, len(A) - 1)] = 1.0

    V = np.zeros((len(S),))
    for indx, v in enumerate(V):
        V[indx] = random.random()

    V[0] = 0
    V[MAX_CELLS - 1] = 0

    while condition:
        delta = 0
        for s in S:
            v = V[s]

            best_a = -1
            best_a_score = None

            for a in A:
                a_score = 0.0
                for s_p in S:
                    for r_idx, r in enumerate(R):
                        a_score += env.transition_probability(s, a, s_p, r_idx) * (r + gamma * V[s_p])
                if best_a_score is None or best_a_score < a_score:
                    best_a = a
                    best_a_score = a_score
                    pi[s, :] = 0.0
                    pi[s, best_a] = 1.0

            V[s] = best_a_score
            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            condition = False

    pi_dict = dict()
    for i, value in enumerate(pi):
        pi_dict[i + 1] = dict(enumerate(value, 1))

    ret_value = dict(enumerate(V.flatten(), 1))

    return pi_dict, ret_value

    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    # TODO
    pass


def policy_evaluation_on_secret_env1() -> ValueFunction:
    """
    Creates a Secret Env1
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    env = Env1()
    pi = np.zeros((len(env.states()), len(env.actions())))

    # pi[:, 1] = 1.0  # Aller toujours à droite ! :)
    # pi[:, 0] = 1.0  # Aller toujours à gauche ! :)
    # pi[:, :] = 0.5  # Aller 50% du temps à gauche et 50% du temps à droite ! :)
    for s in env.states():
        pi[s, 1] = 1.0
    # {1: -1.0000000298023224, 2: -0.5, 3: -0.5, 4: -0.5, 5: 0.0}
    theta = 0.0001
    V = np.zeros((len(env.states()),))
    gamma = 1.0


    while True:
        delta = 0
        for s in env.states():
            v = V[s]
            V[s] = 0
            for a in env.actions():
                for s_p in env.states():
                    for r_idx, r in enumerate(env.rewards()):
                        V[s] += pi[s, a] * env.transition_probability(s, a, s_p, r_idx) * (r + gamma * V[s_p])
            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            break

    return dict(enumerate(V.flatten(), 1))



def policy_iteration_on_secret_env1() -> PolicyAndValueFunction:
    """
    Creates a Secret Env1
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    env = Env1()
    A = env.actions()
    S = env.states()
    R = env.rewards()

    pi = np.zeros((len(env.states()), len(env.actions())))
    for s in S:
        pi[s, random.randint(0, len(env.actions()) - 1)] = 1.0

    V = np.zeros((len(env.states()),))

    while True:
        # 2 : Policy Evaluation

        theta = 0.000001
        gamma = 0.99999

        while True:
            delta = 0
            for s in env.states():
                v = V[s]
                V[s] = 0
                for a in env.actions():
                    for s_p in env.states():
                        for r_idx, r in enumerate(env.rewards()):
                            V[s] += pi[s, a] * env.transition_probability(s, a, s_p, r_idx) * (r + gamma * V[s_p])
                delta = max(delta, abs(v - V[s]))

            if delta < theta:
                break

        # 3 : Policy Improvement
        policy_stable = True
        for s in env.states():
            old_state_policy = np.copy(pi[s, :])

            best_a = -1
            best_a_score = None

            for a in env.actions():
                a_score = 0.0
                for s_p in env.states():
                    for r_idx, r in enumerate(env.rewards()):
                        a_score += env.transition_probability(s, a, s_p, r_idx) * (r + gamma * V[s_p])
                if best_a_score is None or best_a_score < a_score:
                    best_a = a
                    best_a_score = a_score
            pi[s, :] = 0.0
            pi[s, best_a] = 1.0
            if not np.array_equal(old_state_policy, pi[s]):
                policy_stable = False
        if policy_stable:
            break
        pi_dict = dict()
        for i, value in enumerate(pi):
            pi_dict[i + 1] = dict(enumerate(value, 1))

        ret_value = dict(enumerate(V.flatten(), 1))

        return pi_dict, ret_value

def value_iteration_on_secret_env1() -> PolicyAndValueFunction:
    """
    Creates a Secret Env1
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Prints the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    env = Env1()
    A = env.actions()
    S = env.states()
    R = env.rewards()

    V = np.zeros((len(S),))
    pi = np.zeros((len(S), len(A)))
    theta = 0.000000001
    gamma = 0.12345

    while True:
        delta = 0.0
        for s in S:
            v = V[s]
            maxStock = []
            for a in A:
                tmp = 0.0
                for s_p in S:
                    for r_idx, r in enumerate(R):
                        tmp += env.transition_probability(s, a, s_p, r_idx) * (r + gamma * V[s_p])
                maxStock.append(tmp)
            V[s] = np.max(maxStock)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break

    for s in S:
        best_a = -1
        best_a_score = None
        for a in A:
            a_score = 0.0
            for s_p in S:
                for r_idx, r in enumerate(R):
                    a_score += env.transition_probability(s, a, s_p, r_idx) * (r + gamma * V[s_p])
            if best_a_score is None or best_a_score < a_score:
                best_a = a
                best_a_score = a_score
        pi[s, :] = 0.0
        pi[s, best_a] = 1.0

        pi_dict = dict()
        for i, value in enumerate(pi):
            pi_dict[i + 1] = dict(enumerate(value, 1))

        ret_value = dict(enumerate(V.flatten(), 1))

        return pi_dict, ret_value


def demo():
    print(policy_evaluation_on_line_world())
    print(policy_iteration_on_line_world())
    print(value_iteration_on_line_world())

    print(policy_evaluation_on_grid_world())
    print(policy_iteration_on_grid_world())
    print(value_iteration_on_grid_world())

    print(policy_evaluation_on_secret_env1())
    print(policy_iteration_on_secret_env1())
    print(value_iteration_on_secret_env1())
