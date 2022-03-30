from typing import Tuple
from single_agent_env import SingleAgentEnv
import  numpy as np

class TicTacToe(SingleAgentEnv):
    def __init__(self, size):
        self.size = size
        self.board = np.zeros(self.size*self.size)
        self.game_over = False
        self.letters_to_move = [0, 1, 2, 3, 4, 5, 6, 7, 8][:self.size * self.size]
        self.curr_pos = -1
        self.you_win = False
        self.other_win = False
        self.game_draw = False
        self.turn = True

    def get_board(self):
        return self.board.reshape([self.size, self.size])

    def state_id(self) -> int:
        return str(self.board)

    def is_game_over(self) -> bool:
        
        draw = np.count_nonzero(self.board) == self.size * self.size
        if (self.board[0] != 0) and (self.board[0] == self.board[1] == self.board[2]):
            if (self.board[0] == 1):
                self.you_win = True
                self.other_win = False
            else:
                self.other_win = True
                self.you_win = False

            self.game_over = True
            return True

        elif (self.board[3] != 0) and  (self.board[3] == self.board[4] == self.board[5]) :
            if (self.board[3] == 1):
                self.you_win = True
                self.other_win = False
            else:
                self.other_win = True
                self.you_win = False
            self.game_over = True
            return True 
        elif (self.board[6] != 0) and (self.board[6] == self.board[7] == self.board[8]):
            if (self.board[6] == 1):
                self.you_win = True
                self.other_win = False
            else:
                self.other_win = True
                self.you_win = False
            self.game_over = True
            return True
        elif (self.board[0] != 0) and (self.board[0] == self.board[3] == self.board[6]):
            if (self.board[0] == 1):
                self.you_win = True
                self.other_win = False
            else:
                self.other_win = True
                self.you_win = False
            self.game_over = True
            return True
        elif (self.board[1] != 0) and (self.board[1] == self.board[4] == self.board[7]):
            if (self.board[1] == 1):
                self.you_win = True
                self.other_win = False
            else:
                self.other_win = True
                self.you_win = False
            self.game_over = True
            return True
        elif (self.board[2] != 0) and (self.board[2] == self.board[5] == self.board[8]) :
            if (self.board[2] == 1):
                self.you_win = True
                self.other_win = False
            else:
                self.other_win = True
                self.you_win = False
            self.game_over = True
            return True
        elif (self.board[0] != 0) and (self.board[0] == self.board[4] == self.board[8] ):
            if (self.board[0] == 1):
                self.you_win = True
                self.other_win = False
            else:
                self.other_win = True
                self.you_win = False
            self.game_over = True
            return True
        elif (self.board[2] != 0) and (self.board[2] == self.board[4] == self.board[6]):
            if (self.board[2] == 1):
                self.you_win = True
                self.other_win = False
            else:
                self.other_win = True
                self.you_win = False
            self.game_over = True
            return True
        else:
            self.you_win = False
            self.other_win = False
            self.game_over = draw
            self.game_draw = draw
            return draw
        

    def act_with_action_id(self, action_id: int):
        if self.turn:
            self.board[self.letters_to_move.index(action_id)] = 1
            self.curr_pos = action_id
            self.turn = not(self.turn)
        else:
            self.board[self.letters_to_move.index(action_id)] = 2
            self.curr_pos = action_id
            self.turn = not(self.turn)

    def act_with_random_act(self):
        available_actions = self.available_actions_ids()
        action = np.random.choice(available_actions,1)[0]
        self.board[self.letters_to_move.index(action)] = 2

    def play_act(self, action, player):
        
        self.board[self.letters_to_move.index(action)] = player
        self.curr_pos = action
    

    def score(self) -> float:
        # if self.you_win:
        #     return 100
        # elif self.other_win:
        #     return -100
        # else:
        sums = self.get_sums_of_board()
        if self.size in sums:
            return 10 - np.count_nonzero(self.board)  # Punish longer games
        elif -self.size in sums:
            return -10 + np.count_nonzero(self.board)
        else:
            return 0

    def get_sums_of_board(self):
        local_board = self.get_board()
        return np.concatenate([local_board.sum(axis=0),  # columns
                               local_board.sum(axis=1),  # rows
                               np.trace(local_board),  # diagonal
                               np.trace(np.fliplr(local_board))], axis=None)  # other diagonal

    def available_actions_ids(self) -> np.ndarray:
        res = []
        for i in range(len(self.board)):
            if self.board[i] == 0:
                res.append(i)
        return np.array(res)

    def reset(self):
        self.curr_pos = 0
        self.board = np.zeros(self.size * self.size)
        self.game_over = False
    def reset_random(self):
        self.curr_pos = np.random.randint(9)
        self.board = np.zeros(self.size * self.size)
        self.game_over = False
    def view(self): 
        newBoard = self.board.reshape((self.size,self.size))
        for i in range(3):
            print("")
            for j in range(3):
                if newBoard[i][j] == 1:
                    print("x", end = " ")
                elif newBoard[i][j] ==2:
                    print("o", end = " ")
                else:
                    print("-", end = " ")
                

