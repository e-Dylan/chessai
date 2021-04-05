#!/usr/bin/env python
import torch
import time
from state import State
from train import Net
import chess

MODEL_PATH = f"models"
MODEL_FILE = "MODEL-e20-s2m.pth"

MAX_VALUE = 10000
SEARCH_DEPTH = 3

class Evaluator():

	values = {
		chess.PAWN: 1,
		chess.KNIGHT: 2,
		chess.BISHOP: 3,
		chess.ROOK: 5,
		chess.QUEEN: 9,
		chess.KING: 0
	}

	def __init__(self):
		self.reset()
		self.memo = {}
	
	def reset(self):
		self.count = 0

	def __call__(self, state):
		self.count += 1
		key = state.key()
		if key not in self.memo:
			self.memo[key] = self.value(state)
		return self.memo[key]
	
	def value(self, state):
		board = state.board

		if board.is_game_over():
			if board.result() == "1-0":
				return MAX_VALUE
			elif board.result() == "0-1":
				return -MAX_VALUE
			else:
				return 0

		# board piece values
		val = 0.0
		pm = state.board.piece_map()
		for x in pm:
			tval = self.values[pm[x].piece_type] # integer piece value
			if pm[x].color == chess.WHITE:
				val += tval
			else:
				val -= tval
		
		# factor in # of legal moves
		current_turn = board.turn
		board.turn = chess.WHITE
		val += (0.1 * board.legal_moves.count())
		board.turn = chess.BLACK
		val -= (0.1 * board.legal_moves.count())
		board.turn = current_turn

		return val


	def computer_minimax(self, state, depth, a, b, big=False):
		if depth >= SEARCH_DEPTH or state.board.is_game_over():
			return self(state)

		turn = state.board.turn
		if turn == chess.WHITE:
			ret = -MAX_VALUE
		else:
			ret = MAX_VALUE

		if big:
			bret = []

		moves = []
		for move in state.board.legal_moves:
			state.board.push(move)
			moves.append((self(state), move))
			state.board.pop()
		moves = sorted(moves, key=lambda x: x[0], reverse=state.board.turn)

		# beam
		if depth >= 3:
			moves = moves[:10]

		for move in [x[1] for x in moves]:
			state.board.push(move)
			tval = self.computer_minimax(state, depth+1, a, b)
			state.board.pop()
			
			if big:
				bret.append((tval, move))

			if turn == chess.WHITE:
				
				ret = max(ret, tval)
				a = max(a, ret)
				if a >= b:
					# 
					break
			else:
				# Black turn
				ret = min(ret, tval)
				b = min(b, ret)
				if a >= b:
					break

		if big:
			return ret, bret
		else:
			return ret

	def search_moves(self, state):
		ret = []
		start = time.time()
		self.reset()
		bval = self(state)
		cval, ret = self.computer_minimax(state, depth=0, a=-MAX_VALUE, b=MAX_VALUE, big=True)
		eta = time.time() - start
		best_moves = sorted(ret, key=lambda x: x[0], reverse=state.board.turn)
		best_move = best_moves[0][1]
		print(f"'{best_move}' ({bval:.2f} -> {cval:.2f}): explored {self.count} nodes in {eta:.3f} seconds {int(self.count/eta)}/sec")
		return best_moves

# ASSUME PLAYER IS WHITE -> add logic to check value based on what color ai is later

if __name__ == "__main__":
	pass
	# play through an entire game
#     state = State()

#     evaluator = Evaluator()
	
#     evaluator.value(state)
#     best_moves = evaluator.search_moves(state)
#     best_move = best_moves[0][0]
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
