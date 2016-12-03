import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial import distance

try:    #initialize the priority Queue
    import Queue as q
except ImportError:
    import Queue as q



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


def find_xy_grid_length(points):
    xdist=[]
    ydist=[]
    threshold = 0.95;
    dist_matrix=[[[0 for i in range(2)] for j in range(len(points))] for k in range(len(points))]
    eu_dist_matrix = [[0 for j in range(len(points))] for k in range(len(points))]
    for i in range(len(points)):
        for j in range(i,len(points)):
            dist_matrix[i][j][0] = math.fabs(points[i][0]- points[j][0])
            dist_matrix[i][j][1] = math.fabs(points[i][1] - points[j][1])
            eu_dist_matrix[i][j] = distance.euclidean(points[i], points[j])
            dist_matrix[j][i] = dist_matrix[i][j]
            eu_dist_matrix[j][i] = eu_dist_matrix[i][j]
            xdist.append(dist_matrix[i][j][0])
            ydist.append(dist_matrix[i][j][0])

    bins = np.linspace(10, 1000, 400)
    bins_y,vals_y,_=plt.hist(ydist,bins)
    bins_x, vals_x, _ = plt.hist(xdist, bins)
    #plt.show()

    for i in range(0, len(bins_y)):
        elem = bins_y[i]
        if elem == bins_y.max():
            break
    lower_y = [val for val in ydist if val < (vals_y[i] / threshold)]
    upper_y = [val for val in lower_y if val > (vals_y[i] * threshold)]
    y_grid_size= np.mean(upper_y)


    for i in range(0, len(bins_x)):
        elem = bins_x[i]
        if elem == bins_x.max():
            break
    lower_x = [val for val in xdist if val < (vals_x[i] / threshold)]
    upper_x = [val for val in lower_x if val > (vals_x[i] * threshold)]
    x_grid_size= np.mean(upper_x)
    grid_size=(x_grid_size + y_grid_size)/2

    return eu_dist_matrix, dist_matrix, grid_size, y_grid_size, x_grid_size

def validate_corner(points):
    dist_matrix, _,grid_size,y_grid,x_grid=find_xy_grid_length(points)
    valid_corners=[]
    threshold=0.95;
    for i in range(len(points)):
        under = [val for val in dist_matrix[i] if val < grid_size*0.2]
        if len(under) < 2 : # distance to itself is 0. anything apart from itself
            below = [val for val in dist_matrix[i] if val < grid_size/threshold]
            neighbours= [val for val in below if val > grid_size*threshold]
            if len(neighbours):
                valid_corners.append(points[i])
    return valid_corners, grid_size,y_grid,x_grid

def synthetic_corner(ref_points): # ref point => [dist, x_coord, y_coord, x_index, y_index])

    new_point=[]
    x_corner_list=np.array([[0.00 for i in range(9)] for j in range (9)])
    y_corner_list = np.array([[0.00 for i in range(9)] for j in range(9)])
    for pt in ref_points:
        x_corner_list[pt[3]][pt[4]]=pt[1]
        y_corner_list[pt[3]][pt[4]] = pt[2]

    i=1
    j=1
    k=1
    while k<19:
        if i<9 and j<9:
            if not x_corner_list[i][j]:
                if i <= 3:
                    if j>3:
                        x_corner_list[i][j] = x_corner_list[i][j - 1] + x_corner_list[i][j - 2] - x_corner_list[i][j - 3]
                        y_corner_list[i][j] = y_corner_list[i][j - 1] + y_corner_list[i][j - 2] - y_corner_list[i][j - 3]
                    else:
                        if i==3:
                            x_corner_list[i][j] = 2*x_corner_list[i - 1][j] - x_corner_list[i - 2][j]
                            y_corner_list[i][j] = 2*y_corner_list[i - 1][j] - y_corner_list[i - 2][j]
                        elif i==2:
                            if  y_corner_list[i][j + 1] and y_corner_list[i][j + 2] and y_corner_list[i][j + 3]:
                                x_corner_list[i][j] = x_corner_list[i][j + 1] + x_corner_list[i][j + 2] - x_corner_list[i][j + 3]
                                y_corner_list[i][j] = y_corner_list[i][j + 1] + y_corner_list[i][j + 2] - y_corner_list[i][j + 3]
                            elif y_corner_list[i][j + 1] and y_corner_list[i][j + 2]:
                                x_corner_list[i][j] = 2*x_corner_list[i][j + 1] - x_corner_list[i][j + 2]
                                y_corner_list[i][j] = 2*y_corner_list[i][j + 1] - y_corner_list[i][j + 2]
                            else:
                                x_corner_list[i][j] =x_corner_list[i+1][j]
                                y_corner_list[i][j] = y_corner_list[i][j+1]
                elif j<= 3:
                    x_corner_list[i][j] = x_corner_list[i - 1][j] + x_corner_list[i - 2][j] - x_corner_list[i - 3][j]
                    y_corner_list[i][j] = y_corner_list[i - 1][j] + y_corner_list[i - 2][j] - y_corner_list[i - 3][j]
                else:
                    x_corner_list[i][j] = (x_corner_list[i - 1][j] + x_corner_list[i - 2][j] - x_corner_list[i - 3][j] +  x_corner_list[i][j - 1] + x_corner_list[i][j - 2] - x_corner_list[i][j - 3]) / 2
                    y_corner_list[i][j] = (y_corner_list[i - 1][j] + y_corner_list[i - 2][j] - y_corner_list[i - 3][j] +  y_corner_list[i][j - 1] + y_corner_list[i][j - 2] - y_corner_list[i][j - 3]) / 2
                new_point.append([0, int(x_corner_list[i][j]), int(y_corner_list[i][j]), i, j])

        if j==1:
            k=k+1
            j=k
            i=1
        else:
            j=j-1
            i=i+1
    for i in range(1,9):
        x_corner_list[i][0] = x_corner_list[i][1] + x_corner_list[i][2] - x_corner_list[i][3]
        y_corner_list[i][0] = y_corner_list[i][1] + y_corner_list[i][2] - y_corner_list[i][3]
        new_point.append([0, int(x_corner_list[i][0]), int(y_corner_list[i][0]), i, 0]) # adding extrapolated points in the x axis
        x_corner_list[0][i] = x_corner_list[1][i] + x_corner_list[2][i] - x_corner_list[3][i]
        y_corner_list[0][i] = y_corner_list[1][i] + y_corner_list[2][i] - y_corner_list[3][i]
        new_point.append([0, int(x_corner_list[0][i]), int(y_corner_list[0][i]), 0, i]) # adding extrapolated points in the y axis

    x_corner_list[0][0] = x_corner_list[0][1] + x_corner_list[0][2] - x_corner_list[0][3]
    y_corner_list[0][0] = y_corner_list[0][1] + y_corner_list[0][2] - y_corner_list[0][3]
    new_point.append([0, int(x_corner_list[0][0]), int(y_corner_list[0][0]), 0, 0]) # adding 0,0

    return new_point

def corner_detect(img_rgb): # returns the list of corners detected from the image
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
    corner, grid_size,y_grid,x_grid=validate_corner(kkk)
    corrected_corner=[];
    for pt in corner:
        corrected_corner.append([pt[0]+ w/2, pt[1]+h/2])
    return corrected_corner, grid_size,y_grid,x_grid


def corner_detector_assisted(img_rgb, ref):
    _, h, _=img_rgb.shape[::-1]
    lower=  math.floor(h/30 )
    higher = math.floor(h*29/30)
    corner=ref[0]

    if corner==1:
        start= [lower , lower]
    elif corner == 3:
        start = [lower, higher]
    elif corner== 2:
        start = [higher, lower]
    elif corner==0: # looks fixed
        start = [higher, higher]
    else:
        start=0
    print ref, ' ', start

    corner, grid_size,y_grid,x_grid= corner_detect(img_rgb)
    dat=[]
    data=[]
    error_flag=0
    i=0
    for pt in corner:
        dist= distance.euclidean(start, pt)
        dat.append([dist, pt[0], pt[1]])
        i=i+1
    dat.sort()
    ref_st=dat[0]
    if distance.euclidean(start, [ref_st[1], ref_st[2]]) < (grid_size*2.7):

        for pt in dat:
            x_val=int(round(math.fabs(ref_st[1]-pt[1])/x_grid)+1)
            y_val = int(round(math.fabs(ref_st[2] - pt[2])/y_grid)+1)
            data.append([pt[0], pt[1], pt[2], x_val, y_val])

        i=0
        for pt in data:
            cv2.circle(img_rgb, (pt[1], pt[2]), 5, (255, 0, 0), -1)
            cv2.putText(img_rgb,str(pt[3])+' '+str(pt[4]),(pt[1], pt[2]),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0, 255, 0),1)
            i=i+1;
        new_data= synthetic_corner(data)
        for pt in new_data:
            if pt[1]>higher or pt[2]> higher or pt[1]< lower or pt[2]< lower:
                error_flag=1;
            cv2.circle(img_rgb, (pt[1], pt[2]), 5, (0, 255, 0), -1)
            cv2.putText(img_rgb,str(pt[3])+' '+str(pt[4]),(pt[1], pt[2]),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0, 0, 255),1)
            i=i+1;
        if error_flag:
            print "\n reconstruction error. frame will be ignored"
            cv2.putText(img_rgb, "RECONSTUCTION ERROR! IGNORING FRAME!!", (int(h/10), int(h/2)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    return img_rgb





