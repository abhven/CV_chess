# this file allows you to detect aruco markers in an image and overlay an object on them
import cv2
import numpy as np
import aruco
import sys

class AR_Marker :
	def __init__(self, marker):
		self.marker = marker
		self.points = np.zeros((4,2), dtype=np.int)
		self.R = np.zeros((3,3))
		cv2.Rodrigues(self.marker.Rvec, self.R)
		self.T = self.marker.Tvec

def detectMarker(img, param_file):
	markers = [];

	detector = aruco.MarkerDetector()
	detector.setMinMaxSize(0.001)

	cam_param = aruco.CameraParameters()
	cam_param.readFromXMLFile(param_file)

	marker_set = detector.detect(img, cam_param, 1.0);

	for marker in marker_set:
		m = AR_Marker(marker)
		for i in range(4):
			m.points[i] = (int(marker[i][0]), int(marker[i][1]))
		markers.append(m)
	
	return markers

def drawMarker(img, marker) :

	colour = ((240,0,0), (0,240,0), (0,0,240), (0,120,120))
	mean_point = [0,0];
	for i in range(4):
		m_point = (int(marker.marker[i][0]), int(marker.marker[i][1]))
		cv2.circle(img, m_point, 2, colour[0])
		mean_point[0] += m_point[0]/4;
		mean_point[1] += m_point[1]/4;

	cv2.putText(img, "id: "+str(marker.marker.id), (int(mean_point[0]), int(mean_point[1])),
				cv2.FONT_HERSHEY_SIMPLEX, 1, (240,0,0),2);

if __name__=="__main__":

	if(len(sys.argv) < 3) :
		print "USAGE: python pokemon_mouseClick.py file.png sample_param.yml"
		exit()

	img_file = sys.argv[1];
	param_file = sys.argv[2];

	frame = cv2.imread(img_file)

	frame = cv2.pyrDown(frame)
	# frame = cv2.pyrDown(frame)

	markers = detectMarker(frame, param_file)

	for marker in markers:
		print marker.marker.id
		for i in range(4):
			i2 = (i+1)%4
			cv2.line(frame, (marker.points[i][0],marker.points[i][1]),
				 (marker.points[i2][0], marker.points[i2][1]), (0,240,0),2)
			drawMarker(frame, marker)


	cv2.imshow('frame', frame)
	cv2.waitKey(0)
