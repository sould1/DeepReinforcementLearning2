from ..do_not_touch.contracts import MDPEnv
import numpy as np
from math import *


class grid_world_mdpEnvv(MDPEnv):
  def __init__(self, size: int):
    self.size = size*size
    self.gridStates = np.arange(size*size)
    self.gridActions = np.array([0, 1, 2, 3])
    self.gridRewards = np.array([-1, 0, 1])
    self.game_over = False
    self.transition_proba = np.zeros((len(self.gridStates), len(self.gridActions), len(self.gridStates), len(self.gridRewards)))
    self.create_T_probability()


  def states(self) -> np.ndarray:
    return self.gridStates

  def actions(self) -> np.ndarray:
    return self.gridActions

  def rewards(self) -> np.ndarray:
    return self.gridRewards

  def is_state_terminal(self) -> bool:
    return self.game_over

  def create_T_probability(self):
    MAX_CELLS = self.size
    # 0 Gauche
    for i in range(0, MAX_CELLS):
      if i != sqrt(MAX_CELLS) - 1 and i != MAX_CELLS - 1:
        if i % int(sqrt(MAX_CELLS)) == 0:
          self.transition_proba[i, 0, i, 1] = 1.0
        else:
          self.transition_proba[i, 0, i - 1, 1] = 1.0

    # 1 Droite
    for i in range(0, MAX_CELLS):
      if i != sqrt(MAX_CELLS) - 1 and i != MAX_CELLS - 1:
        if i + 1 % int(sqrt(MAX_CELLS)) == 0:
          self.transition_proba[i, 1, i, 1] = 1.0
        else:
          self.transition_proba[i, 1, i + 1, 1] = 1.0

    self.transition_proba[int(sqrt(MAX_CELLS) - 1), 1, int(sqrt(MAX_CELLS)), 0] = 1.0
    self.transition_proba[MAX_CELLS - 2, 1, MAX_CELLS - 1, 2] = 1.0

    # 2 Haut
    for i in range(0, MAX_CELLS):
      if i != sqrt(MAX_CELLS) - 1 and i != MAX_CELLS - 1:
        if i < int(sqrt(MAX_CELLS)):
          self.transition_proba[i, 2, i, 1] = 1.0
        else:
          self.transition_proba[i, 2, i - int(sqrt(MAX_CELLS)), 1] = 1.0

    print("sqrt", sqrt(MAX_CELLS) * 2 - 1)
    self.transition_proba[int(sqrt(MAX_CELLS) * 2 - 1), 2, int(sqrt(MAX_CELLS) - 1), 0] = 1.0

    # 3 Bas
    for i in range(0, MAX_CELLS):
      if i != sqrt(MAX_CELLS) - 1 and i != MAX_CELLS - 1:
        if i > MAX_CELLS - int(sqrt(MAX_CELLS)) - 1:
          self.transition_proba[i, 3, i, 1] = 1.0
        else:
          self.transition_proba[i, 3, int(i + sqrt(MAX_CELLS)), 1] = 1.0

    self.transition_proba[int(sqrt(MAX_CELLS) * (sqrt(MAX_CELLS) - 1)) - 1, 3, MAX_CELLS - 1, 2] = 1.0

  def transition_probability(self, s: int, a: int, s_p: int, r: float) -> float:
    return self.transition_proba[s, a, s_p, r]

  def is_state_terminal(self, s: int) -> bool:
    if s == (self.size - 1) or s == (self.size*self.size - 1):
      self.game_over = True
      return self.game_over

  def view_policy(self, ):
    pass












