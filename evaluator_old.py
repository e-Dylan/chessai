#!/usr/bin/env python

MODEL_PATH=f"models"
MODEL_FILE = "MODEL-e20-s2m.pth"
import torch
from state import State
from train import Net
import chess

class Evaluator():
    def __init__(self):
        self.model = Net()
        self.model.load_state_dict(torch.load(f"{MODEL_PATH}/{MODEL_FILE}", map_location=torch.device('cpu')))
    
    def __call__(self, state):
        board = state.serialize()[None]
        output = self.model(torch.tensor(board).float())
        return output.item()
    
# ASSUME PLAYER IS WHITE -> add logic to check value based on what color ai is later

if __name__ == "__main__":
    # Fresh board
    state = State()

    # evaluator = Evaluator()
    # value = evaluator(state)

  #  while not state.board.is_checkmate(): 
  #      # legal moves
  #      best_move = None
  #      best_value = None
  #      for move in state.edges():
  #          state.board.push(move)
  #          value = evaluator(state)
  #          # assume ai is black, best move is smallest (-1 score)
  #          if best_move is None or value < best_value:
  #              best_value = value
  #              best_move = move
  #
  #          state.board.pop()
  #
  #      print(best_move)
  #      print(best_value)
  #      # now that we have the best move - play it.
  #      state.board.push(best_move)
  #      print(state.board)
