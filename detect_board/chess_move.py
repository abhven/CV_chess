import numpy as np
import cv2
import time
import sys
import copy

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

change_thresh = 0.18

def startPoints(p_removed,notp_placed,heatmap_placed):
    totalscore = {}
    for i in p_removed:
        x = i[0]
        y = i[1]
        notp_placed_copy = copy.copy(notp_placed)

        # R placed one step
        # Find whta is being passed on
        score = [0,0,0]
        n = 1
        while len(notp_placed_copy) > 0:
            for j in range(x-n,x+n+1):
                for k in range(y-n,y+n+1):
                    try:
                        if heatmap_placed[j,k] > 0:
                            score[n] = score[n] + heatmap_placed[j,k]
                            if (j, k, heatmap_placed[j, k]) in notp_placed_copy:
                                notp_placed_copy.remove((j, k, heatmap_placed[j, k]))
                    except IndexError:
                        score[n] = score[n]
            n = n + 1
            print n
        if n == 1:
            totalscore[i] = - i[2]
        else:
            totalscore[i] = (sum(score)/n) - (n*heatmap_placed[x,y]) - i[2]
    return (totalscore)

def endPoints(p_placed,notp_removed,heatmap_removed):
    totalscore = {}

    # Have to save notp_removed
    notp_removed_copy = copy.copy(notp_removed)
    for i in p_placed:
        x = i[0]
        y = i[1]
        notp_removed_copy = copy.copy(notp_removed)

        # R placed one step
        # Find whta is being passed on
        score = [0,0,0]
        n = 0
        while len(notp_removed_copy) > 0:
            for j in range(x-n,x+n+1):
                for k in range(y-n,y+n+1):
                    try:
                        if heatmap_removed[j,k] < 0:
                            score[n] = score[n] + heatmap_removed[j,k]
                            if (j,k,heatmap_removed[j,k]) in notp_removed_copy:
                                notp_removed_copy.remove((j,k,heatmap_removed[j,k]))
                    except IndexError:
                        pass
            n = n + 1
        if n == 0:
            totalscore[i] = i[2]
        else:
            totalscore[i] = i[2] - (sum(score)/n)
    return (totalscore)

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
            heatmap_red[i,j] = diff[0]
            heat_red =  (val_red - colour_min) /(colour_max - colour_min)* 255;
            m_red[i*cell_size:(i+1)*cell_size,j*cell_size:(j+1)*cell_size,:] = [255-heat_red,0,heat_red]
            text = '%s:%.2f' % (index,val_red)
            cv2.putText(m_red, text, (j*cell_size, i*cell_size + 30), cv2.FONT_HERSHEY_PLAIN, 0.7, (0,0,0))

            #create the black heat map
            val_black = constrain(diff[1], colour_min, colour_max);
            heatmap_black[i,j] = diff[1]
            heat_black = (val_black - colour_min) / (colour_max - colour_min) * 255;
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


## This function will compute all possible moves as obtained from the heatmaps and the turn of a person
def computeAllPossibleMoves(heatmap_red, heatmap_black, turn):

    ## compute all set of additions and subtractions using the change in heatmap
    ##=========================================================================
    bcell_removed = np.where(heatmap_black < -1 * change_thresh)
    bcell_placed = np.where(heatmap_black > change_thresh)
    rcell_removed = np.where(heatmap_red < -1 * change_thresh)
    rcell_placed = np.where(heatmap_red > change_thresh)
    b_removed = [(x, y,heatmap_black[x,y]) for x, y in zip(bcell_removed[0], bcell_removed[1])]
    b_placed = [(x, y,heatmap_black[x,y]) for x, y in zip(bcell_placed[0], bcell_placed[1])]
    r_removed = [(x, y,heatmap_red[x,y]) for x, y in zip(rcell_removed[0], rcell_removed[1])]
    r_placed = [(x, y,heatmap_red[x,y]) for x, y in zip(rcell_placed[0], rcell_placed[1])]
    print 'black removed = ' + str(b_removed)
    print 'black placed = ' + str(b_placed)
    print 'red removed = ' + str(r_removed)
    print 'red placed = ' + str(r_placed)
    ##=========================================================================
    bcell_start = startPoints(b_removed, r_placed, heatmap_red)
    bcell_end = endPoints(b_placed, r_removed, heatmap_red)

    print bcell_start
    print bcell_end


    all_moves = None

    if turn == 'b':
        # TODO think of how red's info could be used as well in case of a piece capture
        # Has been used in the top
        # TODO: Find all possible moves from Chessnut

        if len(b_removed)>0 and len(b_placed)>0 :
            b_start = [(let[i]+num[j], heatmap_black[i,j])  for (i,j) in b_removed]
            b_stop = [(let[i]+num[j], heatmap_black[i,j])  for (i,j) in b_placed ]
            all_moves = [(x[0],y[0],-x[1]*y[1]) for x in b_start for y in b_stop]
        pass
    elif turn == 'w':
        if len(r_removed) > 0 and len(r_placed) > 0:
            r_start = [(let[i] + num[j], heatmap_red[i, j]) for (i, j) in r_removed]
            r_stop = [(let[i] + num[j], heatmap_red[i, j]) for (i, j) in r_placed]
            all_moves = [(x[0], y[0], -x[1] * y[1]) for x in r_start for y in r_stop]
        pass

    return all_moves

USE_DUMP = True

## this function will use current and previous states to generate the next legal move
def detectMove(cur_board_features, prev_board_features, chessgame):
    print(chessgame)
    heatmap_red, heatmap_black = computeHeatMap(cur_board_features, prev_board_features)

    #TODO determine whose move is it using the chessgame state
    print computeAllPossibleMoves(heatmap_red, heatmap_black, 'b')
    print 'All Possible moves using heatmap difference alone'

    if not USE_DUMP and False:
        object = [cur_board_features, prev_board_features, chessgame]
        f = open('heatmap.pckl', 'wb')
        pickle.dump(object, f)
        f.close()

    #TODO detect all moves in the chess-board which have a high probability

    #TODO check if the moves that you detected using CV has a high correspondence with the leagal moves

    #TODO return the most likely move and update the chess_game accordingly OR Flag an Error

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
                        # print index +' : ' + scores
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


