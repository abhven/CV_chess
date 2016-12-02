import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

def cluster(list,dist_threshold):
    bin=[]
    set=[]
    i=1
    for pt in list:
        found=0
        j=0
        for k in bin:
            dst = distance.euclidean(k, pt)
            if dst<dist_threshold:
                set.append([j,pt])
                found=1
            j=j+1
        if found==0:
            bin.append([pt])
            i=i+1

    set.sort()
    j=0
    xsum=0
    ysum=0
    i=0
    for pt in set:
        if(pt[0]==j):
            i = i + 1
            xsum=xsum+ pt[1][0]
            ysum=ysum+ pt[1][1]

        else:
            if i:
                bin[j]=[(xsum/i, ysum/i)]
                xsum =0
                ysum = 0
                j=j+1
                i=0
    return bin

def find_grid_length(points):
    dist=[]
    threshold = 0.95;
    dist_matrix=[[0 for k in range(len(points))] for j in range(len(points))]
    for i in range(len(points)):
        for j in range(i,len(points)):
            dist_matrix[i][j] = distance.euclidean(points[i], points[j])
            dist_matrix[j][i] = dist_matrix[i][j]
            dist.append(dist_matrix[i][j])

    bins = np.linspace(100, 1100, 400)
    y,x,_=plt.hist(dist,bins)
    #plt.show()
    for i in range(0, len(y)):
        elem = y[i]
        if elem == y.max():
            break
    lower = [val for val in dist if val < (x[i] / threshold)]
    upper = [val for val in lower if val > (x[i] * threshold)]
    grid_size= np.mean(upper)
    return dist_matrix, grid_size

def validate_corner(points):
    dist_matrix, grid_size=find_grid_length(points)
    valid_corners=[]
    threshold=0.95;
    for i in range(len(points)):
        under = [val for val in dist_matrix[i] if val < grid_size*0.2]
        if len(under) < 2 : # distance to itself is 0. anything apart from itself
            below = [val for val in dist_matrix[i] if val < grid_size/threshold]
            neighbours= [val for val in below if val > grid_size*threshold]
            if len(neighbours):
                valid_corners.append(points[i])
    return valid_corners


def corner_detector_basic(img_rgb):
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('type1_corner.png', 0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)
    list = zip(*loc[::-1])
    new_loc = cluster(list, 20)
    kkk = zip(*new_loc[::-1])[0]

    for pt in kkk:
        # cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        cv2.circle(img_rgb, (pt[0] + w / 2, pt[1] + h / 2), 10, (255, 0, 0), -1)

    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('type2_corner.png', 0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)
    list = zip(*loc[::-1])
    new_loc = cluster(list, 20)
    kkk = zip(*new_loc[::-1])[0]
    for pt in kkk:
        # cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,0), 2)
        cv2.circle(img_rgb, (pt[0] + w / 2, pt[1] + h / 2), 10, (0, 255, 0), -1)

    return img_rgb

def corner_detector_combined(img_rgb):
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY) # convert to gray scale

    template1 = cv2.imread('type1_corner.png', 0) #type 1 corner
    res1 = cv2.matchTemplate(img_gray, template1, cv2.TM_CCOEFF_NORMED)

    template2 = cv2.imread('type2_corner.png', 0)
    res2 = cv2.matchTemplate(img_gray, template2, cv2.TM_CCOEFF_NORMED)

    w, h = template1.shape[::-1] # both the templates have the same

    threshold = 0.8

    loc1 = np.where(res1 >= threshold)
    loc2 = np.where(res2 >= threshold)
    list1 = zip(*loc1[::-1])
    list2 = zip(*loc2[::-1])
    list=list1+list2
    new_loc = cluster(list, 20)
    kkk = zip(*new_loc[::-1])[0]
    corner=validate_corner(kkk)
    for pt in corner:
        # cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        cv2.circle(img_rgb, (pt[0] + w / 2, pt[1] + h / 2), 10, (255, 0, 0), -1)


    return img_rgb