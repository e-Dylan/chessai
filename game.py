#!/usr/bin/env python

from state import State
from evaluator import Evaluator
from controller import Controller
from utils.helpers import get_square, get_index
import pygame as pg
import pygame.time
import chess
import time
import random
import cv2
import chess.pgn
pgn = open("data/games.pgn")

IMAGES = {}
WIDTH = HEIGHT = 700
DIMENSION = 8
SQUARE_SIZE = WIDTH//DIMENSION # width / #squares
MAX_FPS = 10

DARK = (81, 108, 196)
LIGHT = (240, 243, 255)

pieces = {"p": "bP", "b": "bB", "r": "bR", "n": "bN", "q": "bQ", "k": "bK", "P": "wP", "B": "wB", "R": "wR", "N": "wN", "K": "wK", "Q": "wQ"}

def load_images():
    pieces = ["bP", "bB", "bR", "bN", "bQ", "bK", "wP", "wB", "wR", "wN", "wK", "wQ"]
    
    for piece in pieces:
        # load pieces, scale to square size.
        IMAGES[piece] = pg.transform.scale(pg.image.load(f"images/{piece}.png"), (SQUARE_SIZE, SQUARE_SIZE))

def valued_moves(state, evaluator): 
	vm = []
	for move in state.possible_moves():
		state.board.push(move)
		vm.append((move, evaluator(state)))
		state.board.pop()
		moves = sorted(vm, key=lambda x: x[1], reverse=state.board.turn)
		print(moves)
	return moves

def is_turn(state, color):
    return state.board.turn == color    

def draw_board(screen):
    colors = [pg.Color("white"), DARK]
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            color = colors[((r+c) % 2)] # 0 or 1 remainder, index color
            pg.draw.rect(screen, color, pg.Rect(c*SQUARE_SIZE, r*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
    
def draw_pieces(screen, state):
	if state.PLAYING_SIDE is chess.WHITE:
		for i in range(len(chess.SQUARES)):
			square = chess.SQUARES[len(chess.SQUARES)-1 - i]
			x = chess.square_file(square)
			y = chess.square_rank(square)
			piece = state.board.piece_at(i)
			if piece is not None:
				bpiece = pieces[piece.symbol()]
			else:
				bpiece = None

			if bpiece != None:
				screen.blit(IMAGES[bpiece], pg.Rect(WIDTH-SQUARE_SIZE-(x*SQUARE_SIZE), y*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
	else:
		for i in range(len(chess.SQUARES)):
			square = chess.SQUARES[i]
			x = chess.square_file(square)
			y = chess.square_rank(square)
			piece = state.board.piece_at(i)
			if piece is not None:
				bpiece = pieces[piece.symbol()]
			else:
				bpiece = None

			if bpiece != None:
				screen.blit(IMAGES[bpiece], pg.Rect(WIDTH-SQUARE_SIZE-(x*SQUARE_SIZE), y*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

class Game():
	def __init__(self):
		pg.init()
		self.screen = pg.display.set_mode((WIDTH, HEIGHT))
		self.clock = pg.time.Clock()
		self.frames = 0
		self.screen.fill(pg.Color("white"))
		self.controller = Controller()
		self.state = State(PLAYING_SIDE=self.controller.PLAYING_SIDE)
		self.evaluator = Evaluator()
		load_images()
		self.running = False
		self.PLAYER_HOLDING = None

	# Start a new chess game with arguments as to game type.
	# Types: [Player vs AI], AI Autoplay, AI vs AI
	def start(self, game_type):
		self.running = True
		while self.running:
			e = pg.event.poll()
			
			# PLAYER VS AI
			if game_type == "pvai":

				# Await game start.
				if not self.state.in_progress:	
					if e.type == pg.QUIT:
						self.running = False
					elif e.type == pg.MOUSEBUTTONDOWN:
						if e.button == 1: # leftclick to be white
							self.state.set_playing_white()
							self.state.start_game()
						elif e.button == 3: # rightclick to be black
							self.state.set_playing_black()
							self.state.start_game()
					continue

				# Await player's move for bot to move.
				if is_turn(self.state, self.state.OPPONENT_SIDE):
					sleep_time = random.uniform(1.2, 2)
					# print(f"sleeping for {sleep_time} seconds before making a move.")
					# time.sleep(sleep_time)
					# find best move and play it
					best_moves = self.evaluator.search_moves(self.state)
					best_move = best_moves[0][1]
					sq1 = best_move.uci()[:2] # g5
					sq2 = best_move.uci()[2:] # g7
					sq1_index = get_index(sq1, self.state.PLAYING_SIDE)
					sq2_index = get_index(sq2, self.state.PLAYING_SIDE)

					print(f"({self.state.OPPONENT_SIDE}) Bot is playing move {move}, index: {sq2_index}, {sq2_index}")
					self.state.board.push(best_move)
					continue

				elif is_turn(self.state, self.state.PLAYING_SIDE): 
					if e.type == pg.MOUSEBUTTONDOWN and e.type == 5:
						location = pg.mouse.get_pos() # (x, y)
						col = location[0]//SQUARE_SIZE
						row = location[1]//SQUARE_SIZE
						# Nothing picked up yet, can pick something up
						if self.PLAYER_HOLDING is None:
							holding_square = get_square((col, row), self.state.PLAYING_SIDE) # returns uci (e.g. 'a1')
							holding_piece = self.state.board.piece_at(chess.square(col, row)) # returns char (e.g. 'r')
							print(holding_piece)
							self.PLAYER_HOLDING = [holding_piece, holding_square];
							print(f"Picked up piece: {self.PLAYER_HOLDING[0]}, square: {self.PLAYER_HOLDING[1]}")
						else:
							try:
								# Holding a piece, put it down
								holding_square = self.PLAYER_HOLDING[1]
								placed_square = get_square((col, row), self.state.PLAYING_SIDE)
								move_string = holding_square+placed_square
								move = chess.Move.from_uci(move_string)
								if move in self.state.board.legal_moves:
									self.state.board.push(move)
								else:
									print("Attempted to make an illegal move, prevented. Move: ", move_string)
								self.PLAYER_HOLDING = None
							except ValueError:
								print("Invalid move was created.")

					print(f"({self.state.PLAYING_SIDE}) Player is playing move {move}")

			# Automatically play against opponents in online chess.
			elif game_type == "autoplay":
				
				# Bot view of board
				view = self.controller.get_analysis()
				view = cv2.resize(view, (550, 550))
				cv2.imshow("better player view", view)
				if cv2.waitKey(25) & 0xFF == ord('q'):
					cv2.destroyAllWindows()
					break
				
				# Await game start.
				if not self.state.in_progress:	
					if e.type == pg.QUIT:
						self.running = False
					
					elif e.type == pg.MOUSEBUTTONDOWN:
						if e.button == 1: # leftclick to be white
							self.state.set_playing_white()
							self.state.start_game()
						elif e.button == 3: # rightclick to be black
							self.state.set_playing_black()
							self.state.start_game()
					continue

				# analyze board every 0.1 seconds
				if self.frames % 1 == 0:
					self.frames = 0
					analysis = self.controller.analyze_board()
					if analysis is not None:
						move = chess.Move.from_uci(analysis[0])

						isOwnMove = analysis[1]
						print(f"got move: {move}, bot's move? {isOwnMove}")
						
						# if opponent's move, calculate + play next move.
						if not isOwnMove:
							# Detected other player's move, add it to the board state.
							self.state.board.push(move)
						else:
							self.state.board.push(move)

					# Bot's turn to move
					if is_turn(self.state, self.controller.PLAYING_SIDE):
						best_move = self.evaluator.search_moves(self.state)[0][1] 
						print("Bot playing best move: ", best_move)
						# Play move in the real game
						self.controller.dragDropFromUci(best_move.uci())

					# Opponent's move
					# if is_turn(self.state, self.controller.OPPONENT_SIDE):
					# 	best_move = self.evaluator.search_moves(self.state)[0][1]
					# 	self.controller.dragDropFromUci(best_move.uci())

				self.frames += 1

			draw_board(self.screen)
			draw_pieces(self.screen, self.state)
			self.clock.tick(MAX_FPS)
			pg.display.flip()

	# Map of indices to uci squares ((0,0) -> a1) based on color at the bottom of the board (player is playing)

if __name__ == "__main__":
	game = Game()
	game.start(game_type="autoplay")


	# FOR AUTOPLAYING SPEEDGAMES

	# pg.init()
	# screen = pg.display.set_mode((WIDTH, HEIGHT))
	# clock = pg.time.Clock()
	# screen.fill(pg.Color("white"))
	# state = State()
	# evaluator = Evaluator()
	# load_images()
	# running = True
	# HOLDING = None

	# # Map of indices to uci squares ((0,0) -> a1) based on color at the bottom of the board (player is playing)

	# while running:

	# 	while not state.board.is_game_over():
	# 		draw_board(screen)
	# 		draw_pieces(screen, state)
	# 		clock.tick(MAX_FPS)
	# 		pg.display.flip()
			
	# 		best_move = valued_moves(state, evaluator)[0]
	# 		move = best_move[0]
	# 		value = best_move[1]
	# 		if move in state.board.legal_moves:
	# 			print(f"T{state.board.turn} PLAYING MOVE: {move} VALUE: {value:.5f}")
	# 			state.board.push(move)

	# 	print("Checkmate. Starting new game...")
		
	# 	state.board.reset()
	# 	if state.PLAYING_SIDE is chess.WHITE:
	# 		state.set_player_black()
	# 	else:
	# 		state.set_player_white()
