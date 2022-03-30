import numpy as np

S = np.zeros(9, int)
S_p = np.zeros(9, int)
A = np.array([1, 2])  # 1: first player, 2: second player
R = np.array([0, 1])
p = np.zeros(S, len(A), S_p, len(R))


def is_win(S: np.array(9)):
    win_scenarios = np.array([0, 1, 2][3, 4, 5][6, 7, 8][0, 3, 6][7, 4, 1][2, 5, 8][0, 4, 8][6, 4, 2])
    for win_pos in win_scenarios:
        if S[win_pos[0]] != 0 and S[win_pos[0]] == S[win_pos[0]] == S[win_pos[0]]:
            return True
        else:
            return False


# 1 first player
for i in range(9):
    if S[i] == 0:
        S_p[i] = 1
    else:
        p[S, 1, S, 0] = 1.0
    if not is_win(S_p):
        p[S, 1, S_p, 0] = 1.0
    else:
        p[S, 1, S_p, 1] = 1.0

# 2 second player
for i in range(9):
    if S[i] == 0:
        S_p[i] = 2
    else:
        p[S, 2, S, 0] = 1.0
    if not is_win(S_p):
        p[S, 2, S_p, 0] = 1.0
    else:
        p[S, 2, S_p, 1] = 1.0
