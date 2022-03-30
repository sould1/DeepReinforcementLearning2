import random
import time

from numba import jit

# from line_world_mdp_definition import *
from grid_world_mdp_definition import *
import random
import time

from numba import jit

# pi = np.zeros((len(S), len(A)))




@jit
def value_iteration(S, A, R, p):
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

    print("MY V: ", V)


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

    return V, pi




start_time = time.time()
V, pi = value_iteration(S, A, R, p)
print(time.time() - start_time)

# print(V)
print("PI ------------------")
print(pi)

print("PI ------------------")
ret_value = dict(enumerate(pi.flatten(), 1))
print(ret_value)
















