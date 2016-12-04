import cv2
import numpy as np
import sys

import colour_sort
import load_colours

# compute the cell score for each of the cell based on colour or any other parameter
# param img_cell :  image of the cell 60x60
# param type :      whether the cell is green or white by default
# param cur_piece : which chess piece was present here
# param colour_set: set of colours of white, green, red, black
def computeCellColourScore(img_cell, type, cur_piece, colour_set):

    # (num_white, num_green, num_black, num_red) = \
    #     computeColourScoreUsingHist(img_cell, type, cur_piece, colour_set)

    [num_white, num_green, num_red, num_black] = colour_sort.colour_sort(img_cell)
    return (num_white, num_green, num_red, num_black)

# one of numerous such functions which are possible
def computeColourScoreUsingHist(img_cell, type, cur_piece, colour_set):

    num_white = 0
    num_green = 0
    num_black = 0
    num_red = 0

    return (num_white, num_green, num_black, num_red)


if __name__=="__main__":

    if(len(sys.argv) < 3) :
        print "USAGE: python detect_features file.png colours.csv"
        exit()

    img_file = sys.argv[1];
    param_file = sys.argv[2];

    all_colours = []
    # all_colours = load_colours.parseCSVMatrix(param_file, 4)
    img = cv2.imread(img_file)

    res = computeCellColourScore(img, 'w', 'p', all_colours)

    outp = cv2.pyrUp(img)
    outp = cv2.pyrUp(outp)
    text1 = 'W: %d, G: %d' % (res[0], res[1])
    text2 = 'R: %d, B: %d' % (res[2], res[3])
    cv2.putText(outp, text1, (10,50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0))
    cv2.putText(outp, text2, (10,90), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0))
    cv2.imshow('output', outp)
    cv2.waitKey(0)

    # print clusterColours(img, 'White', all_colours)
