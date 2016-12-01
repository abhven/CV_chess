import cv2
import numpy as np
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
    res= res1+res2

    print res1
    print 1234
    print res
    w, h = template1.shape[::-1] # both the templates have the same

    threshold = 0.8

    loc = np.where(res1 >= threshold)
    list = zip(*loc[::-1])
    new_loc = cluster(list, 20)
    kkk = zip(*new_loc[::-1])[0]

    for pt in kkk:
        # cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        cv2.circle(img_rgb, (pt[0] + w / 2, pt[1] + h / 2), 10, (255, 0, 0), -1)


    return img_rgb