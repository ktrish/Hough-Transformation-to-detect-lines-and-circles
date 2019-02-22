import numpy as np
import cv2 as cv2
import math
#Sobel Function for edge detection
def sobel(img,x_kernel,y_kernel):
    gx=np.zeros(img.shape)
    gy=np.zeros(img.shape)
    x_image = np.pad(img, pad_width=1, mode='constant', constant_values=0)
    rows=len(x_image)
    cols=len(x_image[0])
    krows=len(x_kernel)
    kcols=len(x_kernel[0])
    for i in range(rows-2):
        for j in range(cols-2):
            result=0
            for r in range(krows):
                for c in range(kcols):
                    result=result+(x_image[i+r][j+c]*x_kernel[r][c])
            gx[i][j]=result

    y_image = np.pad(img, pad_width=1, mode='constant', constant_values=0)
    for i in range(rows-2):
        for j in range(cols-2):
            result=0
            for r in range(krows):
                for c in range(kcols):
                    result=result+(y_image[i+r][j+c]*y_kernel[r][c])
            gy[i][j]=result

    g=np.sqrt(np.add(np.square(gx),np.square(gy)))
    g[g<0]=0
    g[g>255]=255
    g=g.astype(np.uint8)
    return g
#Reading the image and applying edge detection functions
img = cv2.imread('hough.jpg',0)
x_kernel_edge=[[-1,0,1],[-2,0,2],[-1,0,1]]
y_kernel_edge=[[1,2,1],[0,0,0],[-1,-2,-1]]
edge_detection_edge=sobel(img,x_kernel_edge,y_kernel_edge)
x_kernel_diag=[[0,1,2],[-1,0,1],[-2,-1,0]]
y_kernel_diag=[[-2,-1,0],[-1,0,1],[0,1,2]]
edge_detection_diag=sobel(img,x_kernel_diag,y_kernel_diag)
edge_detection=np.maximum(edge_detection_edge,edge_detection_diag)
####Used canny only for testing#####
#edge_detection = cv2.Canny(img,50,150,apertureSize = 3)
cv2.imwrite('edges_detected.jpg',edge_detection)
#######Defining the rho and theta values #######
w, h = img.shape
len_diag = np.ceil(np.sqrt(w * w + h * h))   # max_dist
rhos = np.linspace(-len_diag, len_diag, len_diag * 2.0)
rhos=np.asarray(rhos)
n_rho=len(rhos)
thetas = np.deg2rad(np.arange(0, 181))
n_theta=len(thetas)

#Defining Parameter space for Hough Space
P=np.zeros((n_rho,n_theta))
##For each edge point, finding the corresponding rho value for each theta and storing
#Votes in the Parameter matrix
for i in range(len(edge_detection)):
    for j in range(len(edge_detection[0])):
        if(edge_detection[i][j]>100):
            for theta in range(n_theta):
                rho = int(round((j * (np.cos(thetas[theta])) + i * (np.sin(thetas[theta])))+len_diag))
                P[rho, theta] =P[rho, theta]+ 1
#Taking the indices which have received the maximum votes
max_vote_index=np.argwhere(P>270)
##Forming a list of rho, theta pairs which have received maximum votes
rho_theta=[]
for i in range(len(max_vote_index)):
    rho_theta.append([rhos[max_vote_index[i][0]],thetas[max_vote_index[i][1]]])
rho_theta=np.asarray(rho_theta)

#drawing red vertical lines in green
img_c= cv2.imread('hough.jpg')
for i in range(len(rho_theta)):
    if(math.degrees(rho_theta[i][1])in range(160,181)):
        a = np.cos(rho_theta[i][1])
        b = np.sin(rho_theta[i][1])
        x0 = a*rho_theta[i][0]
        y0 = b*rho_theta[i][0]
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img_c,(x1,y1),(x2,y2),(0,255,0),2)
cv2.imwrite('red_line.jpg',img_c)

#Taking the indices which have received the maximum votes for blue lines
max_vote_index=np.argwhere(P>205)
##Forming a list of rho, theta pairs which have received maximum votes
rho_theta=[]
for i in range(len(max_vote_index)):
    rho_theta.append([rhos[max_vote_index[i][0]],thetas[max_vote_index[i][1]]])
rho_theta=np.asarray(rho_theta)
#drawing blue lines in green
img_c= cv2.imread('hough.jpg')
for i in range(len(rho_theta)):
    if(math.degrees(rho_theta[i][1])in range(144,146)):
        a = np.cos(rho_theta[i][1])
        b = np.sin(rho_theta[i][1])
        x0 = a*rho_theta[i][0]
        y0 = b*rho_theta[i][0]
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img_c,(x1,y1),(x2,y2),(0,255,0),2)
cv2.imwrite('blue_lines.jpg',img_c)

########Coin Detection############
###Defining the possible range of the center coordinates
a = np.linspace(-h, h, 2*h)
a=a.astype(int)
n_a=len(a)
b = np.linspace(-w, w, 2*w)
b=b.astype(int)
n_b=len(b)
r=23
#Parameter space
P=np.zeros((n_a,n_b))
#Computing circles in Hough space for each edge point
for i in range(len(edge_detection)):
    for j in range(len(edge_detection[0])):
        if(edge_detection[i][j]>48):
            for theta in range(n_theta):
                x0=int(j-r*math.cos(thetas[theta]))+h #To make the values positive,
                y0=int(i-r*math.sin(thetas[theta]))+w #h and w are added
                if((x0+r<2*h)and(y0+r<2*w)):
                    P[x0, y0] =P[x0, y0]+ 1
#Taking the points with max votes based on threshold
max_vote_index=np.argwhere(P>175)  #Threshold value
#Taking the center co-ordinates based on the indices returned from maximum voting
points=[]
for i in range(len(max_vote_index)):
    points.append([a[max_vote_index[i][0]],b[max_vote_index[i][1]]])
points=np.asarray(points)
####Drawing circles based on the points#########
img_c= cv2.imread('hough.jpg')
for (x, y) in points:
    cv2.circle(img_c, (x, y), r, (0, 255, 0), 2)
cv2.imwrite('coin.jpg',img_c)
