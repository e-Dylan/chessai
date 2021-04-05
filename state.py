import chess
import numpy as np
import torch
import os
# from train import Net
MODEL_PATH = f"models/"
MODEL_FILE = f"MODEL-e20-s2m.pth"

# Board().find_move(from_square, to_square, promotion[optional]) -> possible useful function for mapping moves (read docs)

class State():
	def __init__(self, board=None, PLAYING_SIDE=None):
		# fen="RNBQKBNR/PPPPPPPP/8/8/8/8/pppppppp/rnbqkbnr w KQkq - 0 1"
		if board is None:
			self.board = chess.Board()
		else:
			self.board = board
						
		self.in_progress = False
		if PLAYING_SIDE:
			self.PLAYING_SIDE = PLAYING_SIDE
			self.OPPONENT_SIDE = not PLAYING_SIDE

	def key(self):
		return(self.board.board_fen(), self.board.turn, self.board.castling_rights, self.board.ep_square)


	def serialize(self):
		# map board pieces to binary state
		assert self.board.is_valid()

		# print(self.board)    

		bstate = np.zeros(64, np.uint8)
		for i in range(64):
			piece = self.board.piece_at(i)
			if piece is not None:
				bstate[i] = {"P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6, # white pieces
					"p": 9, "n": 10, "b": 11, "r": 12, "q": 13, "k": 14}[piece.symbol()]   # black pieces

		# black castling rights -> queenside for white
		if self.board.has_queenside_castling_rights(chess.WHITE):
			assert bstate[0] == 4 # check if topleft square is rook
			bstate[0] = 7
		
		# black castling rights -> kingside for white
		if self.board.has_kingside_castling_rights(chess.WHITE):
			assert bstate[7] == 4
			bstate[7] = 7

		# white castling rights -> queenside for white
		if self.board.has_queenside_castling_rights(chess.BLACK):
			assert bstate[56] == 8+4 # check if topleft square is rook
			bstate[56] = 8+7

		# white castling rights -> kingside for white
		if self.board.has_kingside_castling_rights(chess.BLACK):
			assert bstate[63] == 8+4
			bstate[63] = 8+7

		# check the board state for an en passant
		if self.board.ep_square is not None:
			assert bstate[self.board.ep_square] == 0
			bstate[self.board.ep_square] = 8 # label en passant

		#[64] -> [8, 8]
		bstate = bstate.reshape(8, 8)

		state = np.zeros((5, 8, 8), np.uint8)

		# 1-4 cols to binary
		state[0] = (bstate>>3)&1
		state[1] = (bstate>>2)&1
		state[2] = (bstate>>1)&1
		state[3] = (bstate>>0)&1

		# 5th col stores turn: 1 == white, 0 == black
		state[4] = (self.board.turn * 1.0)

		return state

	def possible_moves(self):
		return list(self.board.legal_moves)

	def set_playing_white(self):
		self.PLAYING_SIDE = chess.WHITE
		self.OPPONENT_SIDE = chess.BLACK
		# Flip board's default FEN to start white on bottom.
		# self.board.apply_transform(chess.flip_vertical)

	def set_playing_black(self):
		self.PLAYING_SIDE = chess.BLACK
		self.OPPONENT = chess.WHITE

	def start_game(self):
		self.in_progress = True

	def end_game(self):
		self.in_progress = False

