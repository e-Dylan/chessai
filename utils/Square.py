import numpy as np
import cv2
import chess
from utils.helpers import pythagSum, getIndices

class Square():
	def __init__(self, image, c1, c2, c3, c4):
		self.c1 = c1
		self.c2 = c2
		self.c4 = c3
		self.c4 = c4

		self.WHITE_VALUE = 726
		self.BLACK_VALUE = 245

		self.contour = np.array([c1,c2,c4,c3], dtype=np.int32)

		M = cv2.moments(self.contour)
		cx = int(M['m10']/M['m00'])
		cy = int(M['m01']/M['m00'])+14

		# center of square
		self.roi = (cx, cy)

		# 0-based index position (x, y)
		self.position = getIndices(self.roi[0], self.roi[1])
		self.radius = 3

		# TR corner, litte bit inwards.
		self.empty_color_rgb = self.roiColor(image, (c3[0]-13, c3[1]+13))
		self.empty_color = self.empty_color_rgb[0]+self.empty_color_rgb[1]+self.empty_color_rgb[2]

	def draw(self, image, color, thickness=2):
		# draw square from 4-corner contour
		ctr = np.array(self.contour).reshape((-1, 1, 2)).astype(np.int32)
		cv2.drawContours(image, [ctr], 0, color, thickness)

	def drawROI(self, image, color, thickness=1):
		cv2.circle(image, self.roi, self.radius, color, thickness)

	# takes image of entire board mask, returns only color for ROI of this square.
	def roiColor(self, image, coords=None):
		mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
		if coords is not None:
			# Getting a specific piece of a square color (empty color)
			cv2.circle(mask, coords, 2, (255, 255, 255), -1)
		else:
			# Getting color of center of the square (piece color)
			cv2.circle(mask, self.roi, self.radius, (255, 255, 255), -1)
		
		# cv2.imshow("ROI mask", image)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()

		average = cv2.mean(image, mask=mask)[::-1]
		# int-inize, return (r, g, b)
		average = (int(average[1]), int(average[2]), int(average[3]))
		return average
	
	def classify(self, image, display=True):
		rgb = self.roiColor(image)
		sum = 0
		for i in range(len(rgb)):
			sum += rgb[i]
		if display:
			cv2.putText(image, str(sum), self.roi, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
		return sum

		# working in rgb sums, not chess bool values -> easier
		# if abs(sum - self.BLACK_VALUE) < 10:
		# 	return chess.BLACK
		# elif abs(sum - self.WHITE_VALUE) < 10:
		# 	return chess.WHITE