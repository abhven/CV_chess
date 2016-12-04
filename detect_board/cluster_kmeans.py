import numpy as np
import cv2
import sys
import load_colours

COLOUR_THRESHOLD = 2

def whiteCloseness(colour, white):
    pass
def blackCloseness(colour, black):
    pass
def greenCloseness(colour, green):
    pass
def redCloseness(colour, red):
    pass

## runs under the assumption that input colours are in the order W,G,R,B
def clusterColours(cell_img, type, inp_colours):
    num_white = 0;
    num_green = 0;
    num_red = 0;
    num_black = 0;

    kmean = {}
    converged = False

    for i in range(len(inp_colours)):
        colour = inp_colours[i][0]
        kmean[colour] = [inp_colours[1:3],np.array([0,0,0]), 0]
        pass

    while not converged:

        ## Assign a label to all colours in the cell_img
        for i in range(cell_img.shape[0]):
            for j in range(cell_img.shape[1]):

                val = cell_img[i][j]

                #check which colour is closest to the given pixel
                white_score = whiteCloseness(val, kmean)
                green_score = greenCloseness(val, kmean)
                red_score = redCloseness(val, kmean)
                black_score = blackCloseness(val, kmean)

                min_score = white_score; min_index = 0;
                scores = (white_score, green_score, red_score, black_score)
                for i in range(1,4):
                    if scores[i] < min_score:
                        min_index = i
                        min_score = scores[i]

                colour = inp_colours[min_index][0]
                kmean[colour][2] += 1
                kmean[colour][1] += val
            pass

        difference = 0

        ## Update the centroid of the cluster
        for i in range(4):
            colour = inp_colours[i][0]
            prev_kmean = kmean[colour][1]
            if kmean[colour][2] > 0:
                kmean[colour][0] = kmean[colour][1]/kmean[colour][2]
                kmean[colour][1] = np.array([0, 0, 0])
                kmean[colour][2] = 0

            difference += np.norm(prev_kmean - kmean[colour][1])


        ## check for the convergence criteria
        if difference < COLOUR_THRESHOLD:
            converged = True

    return (num_white, num_green, num_black, num_red)
