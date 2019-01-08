import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math

img1 = cv.imread("C:/Users/Bumuthu Dilshan/Desktop/Assignment 2/img5.ppm", cv.IMREAD_COLOR)
img2 = cv.imread("C:/Users/Bumuthu Dilshan/Desktop/Assignment 2/img1.ppm", cv.IMREAD_COLOR)

f,axarr = plt.subplots(1,2)
axarr[0].imshow(img1)
axarr[1].imshow(img2)
plt.show()
points=[[(319,154),(180,136) ],[(375,573),(261,519)],[(490,185),(673,110)],[(495,455),(626,408)]] #goes as x1
a=np.zeros((8,9))
i=0
for k in points:
    a[i,:]=np.array([-k[0][0],-k[0][1],-1,0,0,0,k[0][0]*k[1][0],k[0][1]*k[1][0],k[1][0]])
    a[i+1, :] = np.array([ 0, 0, 0,-k[0][0], -k[0][1], -1, k[0][0] * k[1][1], k[0][1] * k[1][1], k[1][1]])
    i=i+2
s,v,d=np.linalg.svd(a)
print("This is d:")
print(d)
hmet=np.reshape(d[8,:],(3,3))
print("This is h metrix:")
print(hmet)

neww=255*np.zeros((800,1200,3))
print(1)

for x in range(img2.shape[0]):
    for y in range(img2.shape[1]):

        xi=np.array([y,x,1])
        xf=np.matmul(hmet,xi)
        xco=math.floor(xf[1]/xf[2])
        yco=math.floor(xf[0]/xf[2])
        if(xco>=1200 or yco>=800):
            continue
        neww[xco+150 , yco+400, :] = img2[x, y, :]



neww=neww.astype('uint8')

cv.cvtColor(img2,cv.COLOR_BGR2RGB)
cv.cvtColor(neww,cv.COLOR_BGR2RGB)


f,axarr=plt.subplots(1,2)
axarr[0].imshow(img2)
axarr[1].imshow(neww)
plt.show()

cv.imshow('dcscio',neww)
cv.waitKey(0)
cv.destroyAllWindows()
