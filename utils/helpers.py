import chess
import math


# Resized board for piece tracking
RESIZE_WIDTH = RESIZE_HEIGHT = 650
RESIZE_SQUARE_SIZE = RESIZE_WIDTH//8

REFRESH_RATE = 2 # seconds

# takes a tuple index board position (0, 1) -> returns uci string ('a1')
def get_square(pos, PLAYING_COLOR):
	x = pos[0]
	y = pos[1]
	if PLAYING_COLOR == chess.WHITE:
		files = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g", 7: "h" }
		rows = {0: "8", 1: "7", 2: "6", 3: "5", 4: "4", 5: "3", 6: "2", 7: "1"}
		file = files[x]
		row = rows[y]
		square = file+row
		return square
	elif PLAYING_COLOR == chess.BLACK:
		files = {0: "h", 1: "g", 2: "f", 3: "e", 4: "d", 5: "c", 6: "b", 7: "a" }
		rows = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5", 5: "6", 6: "7", 7: "8"}
		file = files[x]
		row = rows[y]
		square = file+row
		return square

# takes uci string of a square (e.g. "a1")
def get_index(uci, PLAYING_COLOR):
	letter = uci[0]
	number = uci[1]
	if PLAYING_COLOR == chess.WHITE:
		files = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5,"g": 6, "h": 7 }
		rows = {"1": 7, "2": 6, "3": 5, "4": 4, "5": 3, "6": 2, "7": 1, "8": 0}
		x = files[letter]
		y = rows[number]
		index = [x, y]
		return index
	elif PLAYING_COLOR == chess.BLACK:
		files = {"a": 7, "b": 6, "c": 5, "d": 4, "e": 3, "f": 2,"g": 1, "h": 0 }
		rows = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6, "8": 7}
		x = files[letter]
		y = rows[number]
		index = [x, y]
		return index

def pythagSum(arr1, arr2):
	sum = 0
	for i in range(0, 3):
		sum += (arr1[i] - arr2[i])**2
	return sum

# Controller class methods

# takes screen coordinates and returns array indices [0]-[7] 
# for the square moved to/from
def getIndices(x, y):
	xIdx = (x//RESIZE_SQUARE_SIZE)
	yIdx = (y//RESIZE_SQUARE_SIZE)
	return (xIdx, yIdx)