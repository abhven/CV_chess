#testing out the harris corners and edge detection on a chess-board image
import cv2
import numpy as np
import sys

import calib
import ar_marker as ar

def getBoardCorners(frame, param_file):

    a = 0.25
    b = 0.2

    board_corners = np.zeros((4,2))

    ar_points = {}
    board_ar_map = {}
    ar_points[471] = [(-a, -a), (-a, 0), (0, 0), (0, -a)]
    board_ar_map[471] = (0, 2)          # the first is the marker index and the second is the corner number
    ar_points[273] = [(-0.2509, 1.6025), (-0.249, 1.8196), (-0.0321, 1.8202), (-0.02980, 1.607)]
    board_ar_map[273] = (3, 3)
    ar_points[298] = [(1.6, -0.25), (1.6, 0), (1.85, 0), (1.85, -0.25)]
    board_ar_map[298] = (1, 1)
    ar_points[983] = [(1.6, 1.6), (1.6, 1.85), (1.85, 1.85), (1.85, 1.6)]
    board_ar_map[983] = (2, 0)
    real_coord = [];
    img_coord = []

    point = np.array([[a], [a], [1]])
    H=[]
    markers = ar.detectMarker(frame, param_file)

    for marker in markers:
        # print marker.marker.id
        if marker.marker.id in ar_points:
            marker_index = board_ar_map[marker.marker.id][0]
            point_index = board_ar_map[marker.marker.id][1]
            # print point_index
            # print marker.points[point_index][0]
            board_corners[marker_index] = np.array((marker.points[point_index][0], marker.points[point_index][1]))

            real_coord.extend(ar_points[marker.marker.id])
            for i in range(4):
                img_coord.append((marker.points[i][0], marker.points[i][1]))

            # for i in range(4):
            #     i2 = (i + 1) % 4
            #     cv2.line(frame, (marker.points[i][0], marker.points[i][1]),
            #              (marker.points[i2][0], marker.points[i2][1]), (0, 240, 0), 2)

            # cv2.line(frame, (0, 0), (marker.points[0][0], marker.points[0][1]), (240, 0, 0), 1)
            # cv2.line(frame, (0, 0), (marker.points[2][0], marker.points[2][1]), (0, 0, 240), 1)

    return board_corners
# H = calib.affineCalibration(img_coord, real_coord);
	# if H != None:
	# 	# print point
	# 	# print H
	# 	proj = np.dot(np.linalg.inv(H), point);
	# 	# print proj
	# 	cv2.line(frame, (0, 0), (proj[0], proj[1]), (0, 240, 0), 1)
    #
    #
	# for i in range(9):
	# 	for j in range(9):
	# 		point = np.array([[i*b], [j*b], [1]])
	# 		proj = np.dot(np.linalg.inv(H), point);
    #
	# 		cv2.circle(frame, (proj[0], proj[1]),2, (0,0,240))
	# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# gray = np.float32(gray)
	# gray = cv2.cornerHarris(gray, 2, 3, 0.14)
	# dst = cv2.dilate(gray, None)
	# edges = cv2.Canny(img, 70, 150)

if __name__=="__main__":

    if(len(sys.argv) < 3) :
        print "USAGE: python detect_features file.png sample_param.yml"
        exit()

    img_file = sys.argv[1];
    param_file = sys.argv[2];

    frame = cv2.imread(img_file)
    frame = cv2.pyrDown(frame)
    frame = cv2.pyrDown(frame)

    board_corners = getBoardCorners(frame, param_file)
    for i in range(board_corners.shape[0]):
        point = board_corners[i]
        cv2.circle(frame, ((int)(point[0]), (int)(point[1])), 3, ((3-i)*240/3, 0, i*240/3), 2)

    cv2.imshow('output', frame)
    cv2.waitKey(0)