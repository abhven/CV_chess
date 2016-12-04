import cv2
import numpy as np
import sys
import time

import detect_corners_r as dc
from corner_detector import *
import cellscore as cs

if __name__=="__main__":

    if(len(sys.argv) < 3) :
        print "USAGE: python video_detection file.mov sample_param.yml"
        exit()

    detection_status = {}
    detection_status['board'] = False
    detection_status['corners'] = False
    frame_index = 0

    img_file = sys.argv[1];
    param_file = sys.argv[2];

    cap = cv2.VideoCapture(img_file)
    ret, frame = cap.read()

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('output.avi', fourcc, 20.0, (692, 692))

    if not ret:
        print 'can\'t open the video'
        exit()

    while ret:

        ##==== Print some basic info for each iteration ====
        frame_index += 1
        print "Frame : " + str(frame_index)

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
        print 'Board Detection :' + str(detection_status['board']) + 'execution time : ' + str(del_t)

        ##====Executing the corner detection and updation =====
        t_start = time.time()
        if detection_status['board']:
            # Display the res_boardult of board detection if successful
            cv2.imshow('output', res_board[0])
            # out.write(res_board[0])
            # cv2.waitKey(0)

            # run the main code for corner detection
            [corner_error_flag, outp_corners,allcorners] = corner_detector_assisted(res_board[0], res_board[4])
            detection_status['corners'] = not corner_error_flag
            ##To see the squares
            if corner_error_flag == False:
                squares = get_squares(res_board[0],allcorners)
                # print(squares)
                # Displays individual squares
                for i in squares:
                    cv2.imshow(i, squares[i])
                    cv2.waitKey(30)
                    cv2.waitKey(0)

            # display the result of corner detection
            cv2.imshow('corners', outp_corners)
            cv2.waitKey(40)

        t_stop = time.time()
        del_t = t_stop - t_start
        print 'Corner Detection :' + str(detection_status['board'])+ '; execution time is ' + str(del_t)

        ##====Extract feature vector for each of the cell =====
        #  (could be number of cells for black/white or number of arcs???

        ##====Use the feature vectors to compute a cell move ====

        ##====Check if the cell is valid ====

        ret, frame = cap.read()

    cv2.waitKey(0)
