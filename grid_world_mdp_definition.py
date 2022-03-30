import numpy as np
from math import *

board = np.zeros( (3,3) )

board = np.zeros(range(9))

MAX_CELLS = 25

# S = np.zeros(shape=(MAX_CELLS, MAX_CELLS),)

S = np.arange(MAX_CELLS)
A = np.array([0, 1, 2, 3])  # 0: Gauche, 1: Droite, 2: Haut, 3: Bas
R = np.array([-1, 0, 1])
p = np.zeros((len(S), len(A), len(S), len(R)))

# 0 Gauche
for i in range(0, MAX_CELLS):
    if i != sqrt(MAX_CELLS) - 1 and i != MAX_CELLS - 1:
        if i % int(sqrt(MAX_CELLS)) == 0:
            p[i, 0, i, 1] = 1.0
        else:
            p[i, 0, i - 1, 1] = 1.0

# 1 Droite
for i in range(0, MAX_CELLS):
    if i != sqrt(MAX_CELLS) - 1 and i != MAX_CELLS - 1:
        if i + 1 % int(sqrt(MAX_CELLS)) == 0:
            p[i, 1, i, 1] = 1.0
        else:
            p[i, 1, i + 1, 1] = 1.0

p[int(sqrt(MAX_CELLS) - 1), 1, int(sqrt(MAX_CELLS)), 0] = 1.0
p[MAX_CELLS - 2, 1, MAX_CELLS - 1, 2] = 1.0

# 2 Haut
for i in range(0, MAX_CELLS):
    if i != sqrt(MAX_CELLS) - 1 and i != MAX_CELLS - 1:
        if i < int(sqrt(MAX_CELLS)):
            p[i, 2, i, 1] = 1.0
        else:
            p[i, 2, i - int(sqrt(MAX_CELLS)), 1] = 1.0

print("sqrt", sqrt(MAX_CELLS)*2 - 1)
p[int(sqrt(MAX_CELLS)*2 - 1), 2, int(sqrt(MAX_CELLS) - 1), 0] = 1.0

# 3 Bas
for i in range(0, MAX_CELLS):
    if i != sqrt(MAX_CELLS) - 1 and i != MAX_CELLS - 1:
        if i > MAX_CELLS - int(sqrt(MAX_CELLS)) - 1:
            p[i, 3, i, 1] = 1.0
        else:
            p[i, 3, int(i + sqrt(MAX_CELLS)), 1] = 1.0

p[int(sqrt(MAX_CELLS)*(sqrt(MAX_CELLS)-1)) - 1, 3, MAX_CELLS - 1, 2] = 1.0


