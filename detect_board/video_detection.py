import cv2
import numpy as np
import sys

import ar_marker as ar
import detect_corners_r as dc
import time
from corner_detector import *

if __name__=="__main__":

    if(len(sys.argv) < 3) :
        print "USAGE: python detect_features file.png sample_param.yml"
        exit()

    img_file = sys.argv[1];
    param_file = sys.argv[2];

    run_detection = False

    # cv2.namedWindow('result1')
    # cv2.namedWindow('STG1_dbg')
    # cv2.namedWindow('result2')
    # cv2.namedWindow('STG2_dbg')

    cap = cv2.VideoCapture(img_file)
    ret, frame = cap.read()

    if not ret:
        print 'can\'t open the video'
        exit()

    while ret:
        if run_detection == False :

            t_start = time.time()
            # cv2.imshow('input', frame)
            res = dc.multiStageDetection(frame, param_file)
            t_stop = time.time()
            del_t = t_stop - t_start
            print "execution time is "+ str(del_t)
            if res[1] > 20:
              #  cv2.imshow('output', cv2.pyrDown(res[0]))
                outp_corners = corner_detector_assisted(res[0],res[4])
                cv2.imshow('corners', outp_corners)
            cv2.waitKey(10)


        if run_detection:
            frame = cv2.pyrDown(frame)
            markers = ar.detectMarker(frame, param_file)
            for marker in markers:
                print marker.marker.id
                for i in range(4):
                    i2 = (i + 1) % 4
                    cv2.line(frame, (marker.points[i][0], marker.points[i][1]),
                             (marker.points[i2][0], marker.points[i2][1]), (0, 240, 0), 2)
                    ar.drawMarker(frame, marker)

            cv2.imshow('output', frame)
            cv2.waitKey(0)

        ret, frame = cap.read()

    cv2.waitKey(0)