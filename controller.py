import pyautogui
import ctypes
import time
import cv2
import math
import numpy as np
import chess
import json
from PIL import ImageGrab
from grabscreen import grab_screen
import matplotlib.pyplot as plt
import requests
from utils.Line import Line
from utils.Square import Square
from utils.helpers import get_square, get_index, pythagSum, getIndices

# STANDARD SCREEN SIZE:
# FULL SCREEN CHROME 1920x1080 MONITOR BIGGEST BOARD SIZE CHESS.COM
# THEN HALF-WIDTH THE WINDOW SIZE

# Real board for self-control

# for max-size sideview window.
# TOPLEFT=(1047-15, 138-15)
# BOTTOMRIGHT=(1879+15, 970+15)

TOPLEFT=(602, 142)
BOTTOMRIGHT=(1411, 954)

WIDTH = HEIGHT = BOTTOMRIGHT[0] - TOPLEFT[0]
SQUARE_SIZE = WIDTH//8

# Resized board for piece tracking
RESIZE_WIDTH = RESIZE_HEIGHT = 650
RESIZE_SQUARE_SIZE = RESIZE_WIDTH//8

class Controller():
		def __init__(self, window_dimensions=(TOPLEFT[0], TOPLEFT[1], BOTTOMRIGHT[0], BOTTOMRIGHT[1])):
			self.window_dimensions = window_dimensions
			self.BOARD_DIMENSIONS = None
			self.corners = None
			self.PLAYING_COLOR, self.OPPONENT_COLOR = None, None
			self.PLAYING_SIDE, self.OPPONENT_SIDE = None, None
			self.init_board()

		def init_board(self):
			self.previous, self.current = None, None
			screen = self.get_screen()

			corners = []
			while len(corners) < 81:
				cleaned, screen = self.clean_image(screen)
				mask = self.init_mask(cleaned, screen)
				edges, color_edges = self.find_edges(mask)
				horizontal, vertical = self.find_lines(edges, color_edges)
				corners = self.find_corners(horizontal, vertical, color_edges)

			self.corners = corners
			# get updated squares
			self.squares, color_edges = self.find_squares(corners, screen)
			self.PLAYING_COLOR = self.detect_bot_color(self.get_screen())
			self.OPPONENT_COLOR = self.detect_opponent_color(self.get_screen())
			# check if piece color is close enough rgb
			if abs(self.PLAYING_COLOR - self.squares[0].WHITE_VALUE) < 50:
				self.PLAYING_SIDE = chess.WHITE
				self.OPPONENT_SIDE = chess.BLACK
			else:
				self.PLAYING_SIDE = chess.BLACK
				self.OPPONENT_SIDE = chess.WHITE

			print("Bot is playing as: ", self.PLAYING_SIDE)
			
		def analyze_board(self):
			screen = self.get_screen()
			# update board vision
			self.previous = self.current
			self.current = screen

			if self.current is not None and self.previous is not None:
				analysis = self.detect_changes(self.previous, self.current)
				return analysis

		def get_screen(self):
			screen = np.array(ImageGrab.grab(bbox=self.window_dimensions))
			screen = cv2.resize(screen, (RESIZE_WIDTH, RESIZE_HEIGHT))
			screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2RGB)
			return screen

		def clean_image(self, image):
			gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

			cleaned = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 125, 1)
			# cv2.imshow("Adaptive Thresholding", cleaned)
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()

			return cleaned, image

		def init_mask(self, cleaned, image):
			# find all closed polygons (each square, entire board, etc.)
			contours, hierarchy = cv2.findContours(cleaned, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

			image_contours = image.copy()

			# check all contours to find the biggest (right) one of the board.
			for c in range(len(contours)):
				area = cv2.contourArea(contours[c])
				per = cv2.arcLength(contours[c], True)

				# filtering the chessboard edge, handling errors from too small contours
				if c == 0:
					Lratio = 0
				if per > 0: # stop divide/0 error
					ratio = area/per
					if ratio > Lratio:
						largest = contours[c]
						Lratio = ratio
						Lper = per
						Larea = area
				else:
					pass

			# draw contours
			# cv2.drawContours(image_contours, [largest], -1, (210,30,30), 3)
			# cv2.imshow("Chess Board Edges", image_contours)
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()
		
			# approximate a polygon to the chessboard contours
			epsilon = 0.1*Lper
			chessboardEdge = cv2.approxPolyDP(largest, epsilon, True)

			# detecting board inside user-set dimensions,
			# come back to this, make mouse operate relative to TL of detected board coordinates

			# save detected chessboard dimensions
			# _, _, boardW, boardH = cv2.boundingRect(chessboardEdge)
			# self.BOARD_DIMENSIONS = (boardW, boardH)
			# w should equal h but if they're a few pixels apart just average it who cares
			# self.SQUARE_SIZE = int(((boardW+boardH)/2) / 8)
			# print(self.SQUARE_SIZE)

			# Create a mask out of chessboard polygon
			mask = np.zeros((image.shape[0], image.shape[1]), 'uint8')*125
			# fill mask polygon white
			cv2.fillConvexPoly(mask, chessboardEdge, 255, 1)
			# copy white pixels into image
			extracted = np.zeros_like(image)
			# fill mask of board space with white
			extracted[mask == 255] = image[mask == 255]
			extracted[np.where((extracted == [125, 125, 125]).all(axis=2))] = [0, 0, 20]

			# cv2.imshow('mask', mask)
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()
		
			return extracted

		def find_edges(self, image):
			# 300, 500 seems to work when: fullscreen chrome, max size chessboard, winkey+rightarrow halfscreen chrome.
			# 200, 400 sometimes works too
			# the threshold is fucking weird
			edges = cv2.Canny(image, 300, 400, None, 3)
			color_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
			# cv2.imshow("edges", edges)
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()
			return edges, color_edges

		def find_lines(self, edges, color_edges):
			lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/800, threshold=300, minLineLength=30, maxLineGap=10)			
			if lines is None:
				print("Failed to find any lines, ensure the chess board is visible in screen. Retrying in 4 seconds")
				return

			a, b, c = lines.shape
			for i in range(a):
				cv2.line(color_edges, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0,255,0), 2, cv2.LINE_AA)

			# cv2.imshow("Lines", color_edges)
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()

			horizontal, vertical = [], []
			for l in range(a):
				[[x1, y1, x2, y2]] = lines[l]
				new_line = Line(x1, x2, y1, y2)
				if new_line.orientation == "horizontal": 
					horizontal.append(new_line)
				elif new_line.orientation == "vertical": 
					vertical.append(new_line)
			
			return horizontal, vertical

		def find_corners(self, horizontal, vertical, color_edges):
			corners = []
			for v in vertical:
				for h in horizontal:
					x, y = v.find_intersection(h)
					corners.append([x, y])

			# remove duplicates
			gcorners = []
			for c in corners:
				matched = False
				for g in gcorners:
					if math.sqrt((g[0]-c[0])*(g[0]-c[0]) + (g[1]-c[1])*(g[1]-c[1])) < 20:
						matched = True
						break
				if not matched:
					gcorners.append(c)

			for g in gcorners:
				cv2.circle(color_edges, (g[0],g[1]), 10, (0,0,255))
				# pass

			# show image with corners circled
			cv2.imshow("Corners", color_edges)
			cv2.waitKey(0)
			cv2.destroyAllWindows()

			return gcorners

		def find_squares(self, corners, color_edges):
			corners.sort(key=lambda x: x[0])
			# 9 rows of corners (8 rows squares)
			rows = [[],[],[],[],[],[],[],[],[]]
			r = 0
			# 81 total corners
			for c in range(0, 81):
				if c > 0 and c % 9 == 0:
					# end of row - new row
					r = r+1

				rows[r].append(corners[c])

			squares = []

			# sort corners by column
			for r in rows:
				r.sort(key=lambda y: y[1])

			# initialize squares
			for r in range(0, 8):
				for c in range(0, 8):
					c1 = rows[r][c]
					c2 = rows[r][c+1]
					c3 = rows[r+1][c]
					c4 = rows[r+1][c+1]

					cv2.circle(color_edges, (c1[0],c1[1]), 10, (0,0,255))
					cv2.circle(color_edges, (c2[0],c2[1]), 10, (0,0,255))
					cv2.circle(color_edges, (c3[0],c3[1]), 10, (0,0,255))
					cv2.circle(color_edges, (c4[0],c4[1]), 10, (0,0,255))

					sq = Square(color_edges, c1, c2, c3, c4)
					sq.draw(color_edges, (0, 0, 255), thickness=2)
					sq.drawROI(color_edges, (255, 0, 0), thickness=2)
					sq.classify(color_edges)
					squares.append(sq)

			# print("Finished detecting board and squares, listening for changes...")

			# cv2.imshow("Squares", color_edges)
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()
			return squares, color_edges
			
		def detect_changes(self, prev, current):
			current_copy = current.copy()

			largestSquare = 0
			secondLargestSquare = 0
			largestDist = 0
			secondLargestDist = 0
			state_change = []

			for sq in self.squares:
				colorPrev = sq.roiColor(prev)
				colorCur = sq.roiColor(current)

				# pythagorean in 3 dims for rgb dist!
				dist = math.sqrt(pythagSum(colorCur, colorPrev))

				if dist > 25:
					# a piece changed on this square.
					state_change.append(sq)
				
				if dist > largestDist:
					secondLargestDist = largestDist
					secondLargestSquare = largestSquare
					largestDist = dist
					largestSquare = sq
				elif dist > secondLargestDist:
					secondLargestDist = dist
					secondLargestSquare = sq

			if len(state_change) == 2:
				# normal move, 1 piece to another square
				sq1, sq2 = state_change[0], state_change[1]

				# check current square piece color
				sq1Curr = sq1.roiColor(current)
				sq2Curr = sq2.roiColor(current)

				# check which square now equals it's empty color
				sumCurr1, sumCurr2 = 0, 0
				distCurr1 = pythagSum(sq1Curr, sq1.empty_color_rgb)
				distCurr2 = pythagSum(sq2Curr, sq2.empty_color_rgb)

				isOwnMove = (self.is_playing_color(sq1, current) or self.is_playing_color(sq2, current))
				
				if distCurr1 < distCurr2:
					# sq1 is now empty (sq1 == from, sq2 == to)
					sqfrom_x, sqfrom_y = sq1.roi[0], sq1.roi[1]
					sqto_x, sqto_y = sq2.roi[0], sq2.roi[1]
					pos_from = getIndices(sqfrom_x, sqfrom_y)
					pos_to = getIndices(sqto_x, sqto_y)
					sq_from = get_square(pos_from, self.PLAYING_SIDE)
					sq_to = get_square(pos_to, self.PLAYING_SIDE)
					print(f"Detected move: {sq_from+sq_to}, Move: {'Bot' if isOwnMove else 'Opponent'}")
					# return the move uci, whether or not it was bot's own move.
					return (sq_from+sq_to, isOwnMove)
				else:
					# sq2 is now empty (sq2 == from, sq1 == to)
					sqfrom_x, sqfrom_y = sq2.roi[0], sq2.roi[1]
					sqto_x, sqto_y = sq1.roi[0], sq1.roi[1]
					pos_from = getIndices(sqfrom_x, sqfrom_y)
					pos_to = getIndices(sqto_x, sqto_y)
					sq_from = get_square(pos_from, self.PLAYING_SIDE)
					sq_to = get_square(pos_to, self.PLAYING_SIDE)
					print(f"Detected move: {sq_from+sq_to}, Move: {'Bot' if isOwnMove else 'Opponent'}")
					return (sq_from+sq_to, isOwnMove)

			# either castling occured or a premove occured
			if len(state_change) == 4:
				# check if two squares contain pieces of bot's colour
				psq = []
				osq = []
				for sq in state_change:
					if self.is_playing_color(sq, current):
						psq.append(sq)
					if self.is_opponent_color(sq, current):
						osq.append(sq)

				# BOARD INDICES ARE COLUMN-BASED (e.g. [7] IS BOTTOM RIGHT CORNER.)
				if len(psq) == 2:
					# bot castled
					if self.PLAYING_SIDE is chess.WHITE:
						# white side castle
						if self.is_playing_color(self.squares[55], current) and self.is_empty(self.squares[39], current) and self.is_empty(self.squares[63], current):
							# castled short right-side, king is on g1
							return ("e1g1", True)
						elif self.is_playing_color(self.squares[23], current) and self.is_empty(self.squares[7], current) and self.is_empty(self.squares[39], current):
							# castled long left-side, king is on c1
							print("Bot (WHITE) castled long. K -> e1c1")
							return ("e1c1", True)
					else:
						# black side castle
						if self.is_playing_color(self.squares[46], current) and self.is_empty(self.squares[31], current) and self.is_empty(self.squares[63], current):
							# castled long right-side, king is on c8
							print("Bot (BLACK) castled short. K -> e8c8")
							return ("e8c8", True)
						elif self.is_playing_color(self.squares[15], current) and self.is_empty(self.squares[31], current) and self.is_empty(self.squares[23], current):
							# castled short left-side, king is on g8
							print("Bot (BLACK) castled long. K -> e8g8")
							return ("e8g8", True)
						
				elif len(osq) == 2:
					# opponent castled
					if self.OPPONENT_SIDE is chess.WHITE:
						# white opponent castle
						if self.is_opponent_color(self.squares[8], current) and self.is_empty(self.squares[0], current) and self.is_empty(self.squares[24], current):
							# castled short right-side, king is on g1
							print("Opponent (WHITE) castled short. K -> e1g1")
							return ("e1g1", False)
						elif self.is_opponent_color(self.squares[40], current) and self.is_empty(self.squares[24], current) and self.is_empty(self.squares[56], current):
							# castled long left-side, king is on c1
							print("Opponent (WHITE) castled long. K -> e1c1")
							return ("e1c1", False)
					else:
						# black opponent castle
						if self.is_opponent_color(self.squares[16], current) and self.is_empty(self.squares[0], current) and self.is_empty(self.squares[32], current):
							# castled long right-side, king is on c8
							print("Opponent (BLACK) castled short. K -> e8c8")
							return ("e8c8", False)
						elif self.is_opponent_color(self.squares[48], current) and self.is_empty(self.squares[32], current) and self.is_empty(self.squares[56], current):
							# castled short left-side, king is on g8
							print("Opponent (BLACK) castled long. K -> e8g8")
							return ("e8g8", False)

		def detect_bot_color(self, screen):
			bot_sq = self.squares[63] # BR square, player side
			return bot_sq.classify(screen, display=False)
		def detect_opponent_color(self, screen):
			bot_sq = self.squares[0] # TL square, opponent side
			return bot_sq.classify(screen, display=False)

		# Detect whether a square's color is equal to a specified color (Check if a square is white/black)
		# Colors are averages and inconsistent so check if within difference of 40
		def is_playing_color(self, sq, image):
			sq_color = sq.classify(image, display=False)
			return abs(sq_color - self.PLAYING_COLOR) < 40
		def is_opponent_color(self, sq, image):
			sq_color = sq.classify(image, display=False)
			return abs(sq_color - self.OPPONENT_COLOR) < 50
		def is_empty(self, sq, image):
			sq_color = sq.classify(image, display=False)
			return abs(sq_color - sq.empty_color) < 50

		def get_analysis(self):
			if self.corners is not None:
				screen = self.get_screen()
				squares, color_edges = self.find_squares(self.corners, screen)

				return color_edges

		def dragDrop(self, startPos, endPos):
			pyautogui.moveTo(startPos)
			time.sleep(0.2)
			pyautogui.dragTo(endPos[0], endPos[1], 0.1, button="left")

		# takes indices [0]-[7] and returns screen coordinate for center
		# of corresponding board square
		def getCoords(self, x, y):
			xPos = TOPLEFT[0] + ((x*SQUARE_SIZE) + (SQUARE_SIZE/2))
			yPos = TOPLEFT[1] + ((y*SQUARE_SIZE) + (SQUARE_SIZE/2))
			return (xPos, yPos)

		def dragDropFromUci(self, move_uci):
			start_idx = get_index(move_uci[:2], self.PLAYING_SIDE)
			end_idx = get_index(move_uci[2:4], self.PLAYING_SIDE)
			start_coords = self.getCoords(start_idx[0], start_idx[1])
			end_coords = self.getCoords(end_idx[0], end_idx[1])
			self.dragDrop(start_coords, end_coords)
			
if __name__ == "__main__":
	controller = Controller()

	while True:
		analysis = controller.get_analysis()
		cv2.imshow("better player view", analysis)
		if cv2.waitKey(25) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break

		move = controller.analyze_board()
		if move is not None:
			print("got move: ", move)
    
	# controller.dragDrop((10, 10), (2359, 769))
	# e2 = controller.getCoords(4, 6)
	# e4 = controller.getCoords(4, 4)
	# pyautogui.moveTo(e2)
	# time.sleep(0.2)
	# pyautogui.dragTo(e4[0], e4[1], 0.5, button="left")
