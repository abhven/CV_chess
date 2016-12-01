#testing out the harris corners and edge detection on a chess-board image
import cv2
import numpy as np
import sys

import ar_marker as ar
from corner_detector import *

warped_board_size = 600
a1_ = 39.2
a2_ = 21.8
b_ = 508.2
offset_ = 45

off_buf = 40
a1 = a1_/b_*warped_board_size
a2 = a2_/b_*warped_board_size
offset = offset_/b_*warped_board_size + off_buf
b = warped_board_size


def getBoardCorners(frame_inp, param_file, debug_mode = 0, draw_mode = 0):

    # a = 1.207*50
    # b = 8*50
    board_corners = np.zeros((4,2))
    R_score = 0
    outp = None
    num_corners = 0;

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

    if real_coord.shape[0] > 1 :
        point = np.array([[[real_coord[1][0], real_coord[1][1]]]])

        if debug_mode :
            print point
            proj = cv2.perspectiveTransform(point, H)
            print proj
            cv2.circle(outp_dbg, (int(proj[0][0][0]), int(proj[0][0][1])), 2, (0, 0, 240))

    if debug_mode:
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

    return [board_corners, outp, R_score, H, outp_dbg]

def multiStageDetection(frame_inp, param_file, debug_mode = 0, draw_mode = 0):

    R_score = 0;
    H = []
    outp_final = []

    res1 = getBoardCorners(frame_inp, param_file, debug_mode, draw_mode)
    if debug_mode :
        print '1:R_score= ' + str(res1[2])
        if draw_mode:
            cv2.putText(res1[4], 'R1score = ' + str(res1[2]), (40, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (240, 0, 0), 2)
            cv2.imshow('STG1_dbg', cv2.pyrDown(res1[4]))

    if res1[1] is not None:

        if res1[2] > 20:
            R_score = res1[2]
            outp_stage1 = res1[1][int(offset):int(2 * a1 + b + offset),
                         int(offset):int(2 * a1 + b + offset)]
            outp_final = outp_stage1
            H = res1[3]
            if draw_mode:
                cv2.imshow('result1', cv2.pyrDown(outp_stage1))
                cv2.waitKey(1)
                # print "STAGE1 complete"
        else:
            res2 = getBoardCorners(res1[1], param_file, 0, 1)
            cv2.putText(res2[4], 'R2score = ' + str(res2[2]), (40, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (240, 0, 0), 2)
            cv2.imshow('STG2_dbg', cv2.pyrDown(res2[4]))

            if res2[1] is not None:
                H2 = np.dot(res2[3], res1[3])
                print ' 2:R_score' + str(res2[2])

                if res2[2] > 20:
                    R_score = res2[2]
                    outp2 = cv2.warpPerspective(frame_inp, H2, (2 * b, 2 * b))
                    outp_stage2 = outp2[offset:int(2*a1 + b + offset),
                                  offset:int(2*a1 + b + offset)]
                    outp_final = outp_stage2
                    H = H2
                    if draw_mode:
                        cv2.imshow('result2', cv2.pyrDown(outp_stage2))
                        cv2.waitKey(1)

    return [outp_final, R_score, H]


if __name__=="__main__":

    if(len(sys.argv) < 3) :
        print "USAGE: python detect_features file.png sample_param.yml"
        exit()

    img_file = sys.argv[1];
    param_file = sys.argv[2];

    frame = cv2.imread(img_file)
    # frame = cv2.pyrDown(frame)
    # frame = cv2.pyrDown(frame)

    res = multiStageDetection(frame, param_file,0,0)

    if res[1] > 20:
        cv2.imshow('result2', cv2.pyrDown(res[0]))
        outp_corners = corner_detector_basic(res[0])
        cv2.imshow('corners', outp_corners)
        cv2.waitKey(0)
