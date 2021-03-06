#testing out the harris corners and edge detection on a chess-board image
import cv2
import numpy as np
import sys
import time

import ar_marker as ar
from corner_detector import *
import cellscore as cs

marker_size = 44.0
chess_cell_size =56.0

warped_board_size = 600
a1_ = 39.2
b_ = 508.2
offset_ = 45

off_buf = 40
epsilon = 1e-3
a1 = a1_/b_*warped_board_size
offset = offset_/b_*warped_board_size + off_buf
b = warped_board_size

print a1

def getBoardCorners(frame_inp, param_file, debug_mode = 0, draw_mode = 0):

    # a = 1.207*50
    # b = 8*50
    board_corners = np.zeros((4,2))
    R_score = 0
    outp = None
    num_corners = 0;
    corner_indexes = [];

    outp_dbg = frame_inp.copy()

    if draw_mode and debug_mode:
        print frame_inp.shape
        # cv2.imshow('debug', frame_inp)
        # cv2.waitKey(0)

    ar_points = {}
    board_ar_map = {}

    ar_points[298] = [[b, 0], [b, -a1], [b+a1, -a1], [ b+a1, 0]] #verified
    board_ar_map[298] = [3, 0]         # the first is the marker index and the second is the corner number
    ar_points[752] = [[-a1, b], [0, b], [0, b+a1], [-a1, b+a1]]  # verified
    board_ar_map[752] = [1, 1]  # the first is the marker index and the second is the corner number
    ar_points[55] = [[0, 0], [-a1, 0], [-a1, -a1], [0, -a1]] # verified
    board_ar_map[55] = [0, 0]
    ar_points[120] = [[b+a1, b], [b+a1, b+a1], [b, b+a1], [b, b]] # verified
    board_ar_map[120] = [2, 3]

    ar_points[303] = [[b/2+a1/2, -a1], [b/2+a1/2, 0], [b/2-a1/2, 0], [b/2-a1/2, -a1]] # verified
    ar_points[532] = [[b/2-a1/2, b+a1], [b/2-a1/2, b], [b/2+a1/2, b], [b/2+a1/2, b+a1]]  # verified
    ar_points[97] = [[-a1, b/2-a1/2], [0, b/2-a1/2], [0, b/2+a1/2], [-a1, b/2+a1/2,]]
    ar_points[442] = [[b+a1, b/2-a1/2], [b+a1, b/2+a1/2], [b, b/2+a1/2], [b, b/2-a1/2]]

    for key in ar_points:
        for elem in ar_points[key]:
            elem[0] += offset + a1
            elem[1] += offset + a1

    # if debug_mode:
        # print board_ar_map
        # print ar_points

    real_coord = np.float32([[0,0]])
    img_coord = np.float32([[0,0]])

    H=[]
    markers = ar.detectMarker(frame_inp, param_file)

    if debug_mode:
        print 'num_markers ' + str(len(markers))

    if draw_mode:
        print 'draw_markers'
        for marker in markers:
            ar.drawMarker(outp_dbg, marker)

    for marker in markers:
        if debug_mode:
            print "detected marker with id " + str(marker.marker.id)

        if marker.marker.id in board_ar_map:
            marker_index = board_ar_map[marker.marker.id][0]
            point_index = board_ar_map[marker.marker.id][1]
            board_corners[marker_index] = np.array((marker.points[point_index][0], marker.points[point_index][1]))
            corner_indexes.append(board_ar_map[marker.marker.id][0])

        if marker.marker.id in ar_points:

            # print "real coordinates id " + str(marker.marker.id)
            real_coord = np.append(real_coord, ar_points[marker.marker.id],0)

            if debug_mode:
                print "marker id is : "+ str(marker.marker.id)
                print real_coord

            for i in range(4):
                img_coord = np.append(img_coord, [[marker.points[i][0], marker.points[i][1]]], 0)

                ## Section of code for deteermining which marker belongs where
                img_point = (marker.points[i][0], marker.points[i][1])
                if debug_mode :
                    print "image coordinates"
                    print [img_point[0], img_point[1]]
                    cv2.circle(outp_dbg, img_point, 2, (0, 0, 240),2)
                #     cv2.imshow('debug', outp_dbg)
                #     cv2.waitKey(0)
                # ---------------------------------------------------------

    if real_coord.shape[0] <= 1:
        return [board_corners, outp, R_score, H, outp_dbg]

    p_mean = np.mean(real_coord[1:], axis=0)
    var_p = np.zeros((2,2))
    n = real_coord.shape[0]-1
    for i in range(1,real_coord.shape[0]):
        p = real_coord[i] - p_mean
        var_p[0][0] += p[0]*p[0]/n
        var_p[0][1] += p[0]*p[1]/n
        var_p[1][0] = var_p[0][1]
        var_p[1][1] += p[1]*p[1]/n
        
    w,v = np.linalg.eig(var_p)
    lambda1 = w[0]
    lambda2 = w[1]
    if(min(lambda1, lambda2) > 0):
        lambda1 = np.sqrt(lambda1)
        lambda2 = np.sqrt(lambda2)
        R_score = lambda2*lambda1/(lambda1 + lambda2)

    if debug_mode:
        print "the variance matrix is "
        print var_p
        print "eigen values are"
        print w
        print "R_score is"
        print R_score

        print "img shape is"
        print img_coord.shape

        print "real shape is"
        print real_coord.shape

        if draw_mode:
            cv2.imshow('debug', outp_dbg)
            cv2.waitKey(0)

    # [H, opt] = cv2.findHomography(img_coord, real_coord);
    # [H, opt] = cv2.findHomography(real_coord[1:], img_coord[1:]);
    [H, opt] = cv2.findHomography(img_coord[1:], real_coord[1:]);

    repoj_error = 0
    for i in range(1,real_coord.shape[0]):
        real_point = np.array([[[real_coord[i][0], real_coord[i][1]]]])
        img_point = img_coord[i,:]
        proj = cv2.perspectiveTransform(real_point, np.linalg.inv(H))

        if debug_mode:
            print real_point
            print proj
            repoj_error += np.linalg.norm(proj-img_point)
        cv2.circle(outp_dbg, (int(proj[0][0][0]), int(proj[0][0][1])), 4, (0, 0, 240),2)
        cv2.circle(outp_dbg, (int(img_point[0]), int(img_point[1])), 4, (240, 0, 0), 2)
    if debug_mode :
        print 'Reprojection error: %f' % (repoj_error)
        if draw_mode:
            cv2.imshow('outp_debg', cv2.pyrDown(outp_dbg))
            cv2.waitKey(0)
        print "H matrix is "
        print H
        print type(H)

    if type(H) is np.ndarray:
        # cv2.imshow('input', frame_inp)
        outp2 = cv2.warpPerspective(frame_inp, H, (2*b,2*b))
        outp = outp2[0:int(off_buf+1.5*b),0:int(off_buf+1.5*b)]
        if draw_mode and debug_mode:
            outp2 = cv2.warpPerspective(outp_dbg, H, (2 * b, 2 * b))
            outp_dbg = outp2[0:int(off_buf + 1.5 * b), 0:int(off_buf + 1.5 * b)]
            cv2.imshow('warped', outp)
            cv2.waitKey(0)

    return [board_corners, outp, R_score, H, outp_dbg, real_coord, corner_indexes]

def multiStageDetection(frame_inp, param_file, debug_mode = 0, draw_mode = 0):

    R_score = 0;
    H = []
    outp_final = []
    board_points = [];
    corner_indexes = [];
    scores = [0,0]
    f = open('R_scores.csv', 'a')

    res1 = getBoardCorners(frame_inp, param_file, debug_mode, draw_mode)
    scores[0] = res1[2]
    if debug_mode :
        print '1:R_score= ' + str(res1[2])
        if draw_mode:
            cv2.putText(res1[4], 'R1score = ' + str(res1[2]), (40, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (240, 0, 0), 2)
            cv2.imshow('STG1_dbg', cv2.pyrDown(res1[4]))
            # cv2.imshow('STG1_dbg', res1[4])

    if res1[1] is not None:

        corner_indexes = res1[6]

        if res1[2] > 40:
            R_score = res1[2]
            outp_stage1 = res1[1][int(offset):int(2 * a1 + b + offset),
                         int(offset):int(2 * a1 + b + offset)]
            outp_final = outp_stage1
            H = res1[3]

            # assign the points which are closest to the board as board_points
            for p in res1[5]:
                p_temp = p - (a1 + offset) * np.array([1, 1])
                p[0] -= offset
                p[1] -= offset
                if ((abs(p_temp[1] - b) < epsilon or abs(p_temp[1]) < epsilon) or
                        (abs(p_temp[0] - b) < epsilon or abs(p_temp[0]) < epsilon)):
                    board_points.append(p)

            if draw_mode:

                for p in board_points:
                    cv2.circle(outp_stage1, (int(p[0]),int(p[1])), 2, (0, 0, 240), 2)
                    print 'drawing circle'
                    print p

                cv2.imshow('result1', cv2.pyrDown(outp_stage1))
                cv2.waitKey(1)


                        # print "STAGE1 complete"
        else:
            res2 = getBoardCorners(res1[1], param_file, debug_mode, draw_mode)
            scores[1] = res2[2]
            if debug_mode:
                print '2:R_score= ' + str(res2[2])
                if draw_mode:
                    cv2.putText(res2[4], 'R2score = ' + str(res2[2]), (40, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (240, 0, 0), 2)
                    cv2.imshow('STG2_dbg', cv2.pyrDown(res2[4]))

            if res2[1] is not None:

                corner_indexes = res2[6]

                H2 = np.dot(res2[3], res1[3])
                if debug_mode:
                    print ' 2:R_score' + str(res2[2])

                if res2[2] > 20:
                    R_score = res2[2]
                    outp2 = cv2.warpPerspective(frame_inp, H2, (2 * b, 2 * b))
                    outp_stage2 = outp2[offset:int(2*a1 + b + offset),
                                  offset:int(2*a1 + b + offset)]
                    outp_final = outp_stage2
                    H = H2
                    # assign the points which are closest to the board as board_points
                    for p in res1[5]:
                        p_temp = p - (a1 + offset) * np.array([1, 1])
                        p[0] -= offset
                        p[1] -= offset
                        if ((abs(p_temp[1] - b) < epsilon or abs(p_temp[1]) < epsilon) and
                                (abs(p_temp[0] - b) < epsilon or abs(p_temp[0]) < epsilon)):
                            board_points.append(p)

                    if draw_mode:

                        for p in board_points:
                            cv2.circle(outp_stage2, (int(p[0]), int(p[1])), 2, (0, 0, 240), 2)
                            print 'drawing circle'
                            print p

                        cv2.imshow('result2', cv2.pyrDown(outp_stage2))
                        cv2.waitKey(1)

    ##TODO remove this later
    corner_indexes = [1,0,2,3]
    f.write('%f,%f\n'%(scores[0], scores[1]))
    f.close()
    return [outp_final, R_score, H, board_points, corner_indexes]


if __name__=="__main__":

    if(len(sys.argv) < 3) :
        print "USAGE: python detect_features file.png sample_param.yml"
        exit()

    img_file = sys.argv[1];
    param_file = sys.argv[2];

    frame = cv2.imread(img_file)

    detection_status = {}
    board_features = {}
    let = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    num = ['1', '2', '3', '4', '5', '6', '7', '8']
    # frame = cv2.pyrDown(frame)
    # frame = cv2.pyrDown(frame)

    ##====Executing the board detection here =====
    t_start = time.time()
    res_board = multiStageDetection(frame, param_file)
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

        # display the result of corner detection
        cv2.imshow('corners', outp_corners)
        cv2.moveWindow('output', 0, 0)
        cv2.moveWindow('corners', 0, 200)
        cv2.waitKey(0)

