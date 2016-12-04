import cv2
import numpy as np

def colour_sort(img):
    img = cv2.blur(img, (5, 5))
    _,w, h = img.shape[::-1]
    colour=[0, 0,0,0] #[ white, green, red, black
    # remove the img variable later for
    rows_W = np.where(img[:,:,1] > 100)
    img[rows_W]=[0,255,0]
    rows_R = np.where(img[:, :, 2] > 100)
    img[rows_R] = [0, 0, 255]
    rows_B = np.where(np.mean(img,2) - np.amin(img,2) < 5)
    img[rows_B] = [255, 0, 0]

    colour[0] = len(rows_W[0])
    colour[2] = len(rows_R[0])
    colour[3] = len(rows_B[0])
    colour[1] = 3600-sum(colour)

    return colour #, img

