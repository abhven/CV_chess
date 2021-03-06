import cv2
import numpy as np
import sys
import time

import chess
import detect_corners_r as dc
from corner_detector import *
import cellscore as cs
import chess_move
from os import path
sys.path.append('../Chessnut')
from Chessnut import Game

let = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
num = ['1', '2', '3', '4', '5', '6', '7', '8']
move_count = 0
def generateStatusMessage():
    status = ''
    status += 'F' + str(frame_index) + ', '

    return status

if __name__=="__main__":

    if(len(sys.argv) < 3) :
        print "USAGE: python video_detection file.mov sample_param.yml"
        exit()

    detection_status = {}
    detection_status['board'] = False
    detection_status['corners'] = False

    engage_detection = False
    frame_index = 0

    board_features = {}
    prev_board_features = {}
    cur_board_features = {}

    cv2.namedWindow('output', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('corners', cv2.WINDOW_AUTOSIZE)
    chessgame = Game()
    print(chessgame)

    img_file = sys.argv[1];
    param_file = sys.argv[2];

    cap = cv2.VideoCapture(img_file)
    ret, frame = cap.read()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 25.0, (1920, 1080))

    if not ret:
        print 'can\'t open the video'
        exit()

    while ret:

        ##==== Print some basic info for each iteration ====
        frame_index += 1
        # input_img = cv2.pyrDown(cv2.pyrDown(frame))
        if frame_index == 1:
            input_img = frame
            input_img[:,:,:] = 0
        img_in = cv2.pyrDown(frame)
        input_img[0:500, 0:900] = img_in[0:500,0:900]
        status_msg = generateStatusMessage()
        cv2.putText(input_img, status_msg, (10, 500), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 0, 255))

        # print "Frame : " + str(frame_index)

        if engage_detection:
            ##====Executing the board detection here =====
            input_img[700:900, 0:500] = 0
            status_msg = 'Key Pressed'
            cv2.putText(input_img, status_msg, (10, 750), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 240, 240), 2)
            status_msg = 'Detecting Game'
            cv2.putText(input_img, status_msg, (10, 850), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 240, 240), 2)
            cv2.imshow('input', input_img)
            cv2.waitKey(5)
            t_start = time.time()
            res_board = dc.multiStageDetection(frame, param_file)
            t_stop = time.time()
            del_t = t_stop - t_start

            detection_status['board'] = False
            detection_status['corners'] = False

            # use threshold to compute success of the stage
            if res_board[1] > 40:
                detection_status['board'] = True
            else:
                detection_status['board'] = False
            print 'Board Detection :' + str(detection_status['board']) + '; Execution time : ' + str(del_t)
            ##============================================

            ##====Executing the corner detection and updation =====
            t_start = time.time()
            if detection_status['board']:
                # Display the res_boardult of board detection if successful
                result_board = cv2.pyrDown(res_board[0])
                rows = result_board.shape[0]
                cols = result_board.shape[1]
                print result_board.shape
                # input_img[0:rows,0:cols] = result_board
                cv2.imshow('output', result_board)
                # out.write(input_img)
                # cv2.waitKey(0)

                #run the main code for corner detection
                [corner_error_flag, outp_corners, all_corners] = fast_corner_detector(res_board[0], res_board[4])
                detection_status['corners'] = not corner_error_flag
                all_colours = []
                # all_colours = load_colours.parseCSVMatrix(param_file, 4)
            t_stop = time.time()
            del_t = t_stop - t_start
            ##============================================
            print 'Corner Detection :' + str(detection_status['corners']) + '; Execution time : ' + str(del_t)

            ##======Generate Cell Features========================
            if detection_status['corners']:
                engage_detection = False
                ## Display the corner results if successful
                input_img[700:900,0:500] = 0
                status_msg = 'Board Corners Detected'
                cv2.putText(input_img, status_msg, (10, 800), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 240, 240), 2)
                cv2.waitKey(5)
                result_board = cv2.pyrDown(outp_corners)
                rows = result_board.shape[0]
                cols = result_board.shape[1]
                input_img[500:500+rows, 554:554+cols] = result_board
                # cv2.imshow('corners', outp_corners)
                squares = get_squares(res_board[0], all_corners)
                # print(squares)
                # Displays individual squares
                for i in squares:
                    pass
                    ## Displaying the individual squares
                    # cv2.imshow(i, squares[i])
                    # print i
                    # cv2.waitKey(30)
                    # cv2.waitKey(0)
                for i in range(8):
                    for j in range(8):
                        index = let[i] + num[j]
                        cell_img = squares[index]

                        # Send correct cell colour and the piece information
                        (W, G, R, B) = cs.computeCellColourScore(cell_img, 'w', 'p', all_colours)
                        pt = all_corners[i][j]
                        # cv2.circle(img_rgb, (pt[0], pt[1]), 5, (0, 255, 0), -1)
                        cv2.putText(outp_corners, str(i) + ' ' + str(j), (pt[0], pt[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1,
                                    (255, 0, 0), 1)
                        red_score = R / 3600.0
                        black_score = B / 3600.0
                        board_features[index] = [red_score, black_score]
                        scores = '%s[%.2f,%.2f]' % (index, red_score, black_score)
                        if (i + j) % 2 == 0:
                            text_color = (220, 40, 40)
                        else:
                            text_color = (200, 240, 240)
                        cv2.putText(outp_corners, scores, (pt[1], pt[0] + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5,
                                    text_color, 1)
                        # print index +' : ' + scores
                        cv2.waitKey(10)
                ##=========================================================

                ##==========Detect a Move on the basis of features=========
                ## compute the current and previous features
                prev_board_features = cur_board_features.copy()
                cur_board_features = board_features.copy()
                if len(cur_board_features) == 64 and len(prev_board_features) == 64:
                    move = chess_move.detectMove(cur_board_features, prev_board_features, chessgame, move_count, input_img)
                    print move
                    # print board_msg
                    # for i in range(len(board_msg)):
                    #     chess_row = board_msg[i]
                    #     cv2.putText(input_img, chess_row, (1400, int((i+1)*50)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5,
                    #                 (0, 240, 240), 1)
                    if move is not None:
                        chessgame.apply_move(move)
                        move_count = move_count + 1
                        status_msg = 'Move %d : %s' % (move_count, move)
                        board_msg_raw = str(chess.Board(str(chessgame)))
                        board_msg = board_msg_raw.split('\n')
                        print board_msg_raw
                    else:
                        status_msg = 'Detection Failed '
                    cv2.putText(input_img, status_msg, (10, 725), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 240, 240), 2)

        ## display the result of corner detection
        out.write(input_img)
        cv2.imshow('input', input_img)
        cv2.moveWindow('input', 0, 0)
        cv2.moveWindow('output', 0, 270)
        # cv2.moveWindow('corners', 0, 400)
        char_input = cv2.waitKey(40)

        if char_input & 0xFF == ord('e'):
            engage_detection = True
        elif char_input & 0xFF == 0x1B:
            break

        ret, frame = cap.read()

    print 'Exiting'
    cap.release()
    cv2.destroyAllWindows()
    out.release()
    cv2.waitKey(100)
