import numpy as np
import cv2
import time
import sys

import cellscore as cs
from corner_detector import *
import detect_corners_r as dc
import pickle
sys.path.append('../Chessnut')
from Chessnut import Game

let = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
num = ['1', '2', '3', '4', '5', '6', '7', '8']

colour_min = -0.6
colour_max = 0.6

def constrain(inp, minval, maxval):
    if inp>maxval:
        inp = maxval
    elif inp<minval:
        inp = minval
    return inp


def computeHeatMap(board_features, prev_board_features):
    size = 480, 480, 3
    cell_size = 60
    m_red = np.zeros(size, dtype=np.uint8)
    m_black = np.zeros(size, dtype=np.uint8)
    heatmap_red = np.zeros((8,8), dtype=np.float32)
    heatmap_black = np.zeros((8, 8), dtype=np.float32)
    for i in range(8):
        for j in range(8):
            index = let[i] + num[j]
            diff = np.array(board_features[index]) - np.array(prev_board_features[index])
            # print 'mat[' + index+ '] diff: ' +str(diff)

            #create the red heat map
            val_red = constrain(diff[0], colour_min, colour_max) ;
            heatmap_red[i,j] = (val_red - colour_min) /(colour_max - colour_min)
            heat_red =  heatmap_red[i,j]* 255;
            m_red[i*cell_size:(i+1)*cell_size,j*cell_size:(j+1)*cell_size,:] = [255-heat_red,0,heat_red]
            text = '%s:%.2f' % (index,val_red)
            cv2.putText(m_red, text, (j*cell_size, i*cell_size + 30), cv2.FONT_HERSHEY_PLAIN, 0.7, (0,0,0))

            #create the black heat map
            val_black = constrain(diff[1], colour_min, colour_max);
            heatmap_black[i,j] = (val_black - colour_min) / (colour_max - colour_min)
            heat_black = heatmap_black[i,j] * 255;
            m_black[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size, :] = [255-heat_black,0,heat_black]
            text = '%s:%.2f' % (index, val_black)
            cv2.putText(m_black, text, (j*cell_size, i*cell_size + 30), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0))

            # print 'vals black :' + str(val_black) + ' red: ' + str(val_red)
            # print 'black :' + str(heat_black) + ' red: ' + str(heat_red)

    cv2.imshow('Red Heatmap', m_red)
    cv2.imshow('Black Heatmap', m_black)
    cv2.moveWindow('Red Heatmap', 640,0)
    cv2.moveWindow('Black Heatmap', 1120, 0)
    cv2.waitKey(0)
    return [heatmap_red, heatmap_black]


USE_DUMP = True

## this function will use current and previous states to generate the next legal move
def detectMove(cur_board_features, prev_board_features, chessgame):
    print(chessgame)
    heatmap_red, heatmap_black = computeHeatMap(cur_board_features, prev_board_features)

    if not USE_DUMP:
        object = [cur_board_features, prev_board_features, chessgame]
        f = open('heatmap.pckl', 'wb')
        pickle.dump(object, f)
        f.close()

    #detect all moves in the chess-board which have a high probability

    #check if the moves that you detected using CV has a high correspondence with the leagal moves

    #return the most likely move and update the chess_game accordingly OR Flag an Error

if __name__=="__main__":

    if USE_DUMP:
        #load data directly using pickle
        f = open(sys.argv[1], 'rb')
        data = pickle.load(f)
        f.close()
        detectMove(data[0], data[1], data[2])

    if(len(sys.argv) < 4) :
        print "USAGE: python detect_features file1.png file2.png sample_param.yml"
        exit()

    chessgame = Game()
    chessgame.apply_move('e2e4')
    chessgame.apply_move('e7e5')
    chessgame.apply_move('g1f3')
    chessgame.apply_move('d7d5')
    chessgame.apply_move('e4d5')
    print(chessgame)

    img_file1 = sys.argv[1];
    img_file2 = sys.argv[2];
    param_file = sys.argv[3];

    frames = []
    frames.append(cv2.imread(img_file1))
    frames.append(cv2.imread(img_file2))

    detection_status = {}
    board_features = {}
    prev_board_features = {}
    cur_board_features = {}

    let = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    num = ['1', '2', '3', '4', '5', '6', '7', '8']
    # frame = cv2.pyrDown(frame)
    # frame = cv2.pyrDown(frame)

    for i in range(2):
        frame = frames[i]
        ##====Executing the board detection here =====
        t_start = time.time()
        res_board = dc.multiStageDetection(frame, param_file)
        t_stop = time.time()
        del_t = t_stop - t_start

        # use threshold to compute success of the stage
        if res_board[1] > 40:
            detection_status['board'] = True;
        else:
            detection_status['board'] = False;
        print 'Board Detection :' + str(detection_status['board']) + ' execution time : ' + str(del_t)

        ##====Executing the corner detection and updation =====
        t_start = time.time()
        if detection_status['board']:
            # Display the res_boardult of board detection if successful
            cv2.imshow('output', res_board[0])
            # out.write(res_board[0])
            # cv2.waitKey(0)

            # run the main code for corner detection
            [corner_error_flag, outp_corners, all_corners] = corner_detector_assisted(res_board[0], res_board[4])
            detection_status['corners'] = not corner_error_flag
            all_colours = []
            # all_colours = load_colours.parseCSVMatrix(param_file, 4)

            ##To see the squares
            if corner_error_flag == False:
                squares = get_squares(res_board[0], all_corners)
                # print(squares)
                # Displays individual squares
                for i in squares:
                    pass
                    # cv2.imshow(i, squares[i])
                    # print i
                    # cv2.waitKey(30)
                    # cv2.waitKey(0)
                for i in range(8):
                    for j in range(8):
                        index = let[i]+num[j]
                        cell_img = squares[index]

                        # Send correct cell colour and the piece information
                        (W,G,R,B) = cs.computeCellColourScore(cell_img, 'w', 'p', all_colours)
                        pt=all_corners[i][j]
                        # cv2.circle(img_rgb, (pt[0], pt[1]), 5, (0, 255, 0), -1)
                        cv2.putText(outp_corners, str(i) + ' ' + str(j), (pt[0], pt[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                    (255, 0, 0), 1)
                        red_score = R/3600.0
                        black_score = B/3600.0
                        board_features[index] = [red_score, black_score]
                        scores = '[%.2f,%.2f]' % (red_score, black_score)
                        if (i+j)%2 ==0:
                            text_color = (220,40,40)
                        else:
                            text_color = (200, 240, 240)
                        cv2.putText(outp_corners, scores, (pt[0], pt[1]-20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5,
                                    text_color, 1)
                        print index +' : ' + scores
                        cv2.waitKey(10)

            ## display the result of corner detection
            cv2.imshow('corners', outp_corners)
            cv2.moveWindow('output', 0, 0)
            cv2.moveWindow('corners', 0,200)
            cv2.waitKey(0)

            ## compute the current and previous features
            prev_board_features = cur_board_features.copy()
            cur_board_features = board_features.copy()
            if len(cur_board_features) == 64 and len(prev_board_features) == 64:
                detectMove(cur_board_features, prev_board_features, chessgame)

