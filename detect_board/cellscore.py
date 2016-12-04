import cv2
import numpy as np

# compute the cell score for each of the cell based on colour or any other parameter
# param img_cell :  image of the cell 60x60
# param type :      whether the cell is green or white by default
# param cur_piece : which chess piece was present here
# param colour_set: set of colours of white, green, black, red
def computeCellColourScore(img_cell, type, cur_piece, colour_set):

    (num_white, num_green, num_black, num_red) = \
        computeColourScoreUsingHist(img_cell, type, cur_piece, colour_set)

    return (num_white, num_green, num_black, num_red)

# one of numerous such functions which are possible
def computeColourScoreUsingHist(img_cell, type, cur_piece, colour_set):
    num_white = 0
    num_green = 0
    num_black = 0
    num_red = 0

    return (num_white, num_green, num_black, num_red)