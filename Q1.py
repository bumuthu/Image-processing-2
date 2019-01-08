import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
img2=cv.imread("C:/Users/Bumuthu Dilshan/Desktop/Assignment 2/img6.ppm")
img1=cv.imread("C:/Users/Bumuthu Dilshan/Desktop/Assignment 2/img2.ppm")
points=[[(450,240),(423,326)],[(352,567),(200,554)],[(583,157),(472,334)],[(100,133),(472,32)]]
A=np.zeros((8, 9))
d=0
f, axarr = plt.subplots(1, 2)
img1cvt=cv.cvtColor(img1,cv.COLOR_BGR2RGB)
img2cvt=cv.cvtColor(img2,cv.COLOR_BGR2RGB)
axarr[0].set_title("First Image")
axarr[0].imshow(img1cvt)
axarr[1].set_title('Second Image')
axarr[1].imshow(img2cvt)
plt.show()
for cor in points:
    A[d, :]=np.array([-cor[0][0], -cor[0][1], -1, 0, 0, 0, cor[0][0] * cor[1][0], cor[0][1] * cor[1][0], cor[1][0]])
    A[d + 1, :] = np.array([0, 0, 0, -cor[0][0], -cor[0][1], -1, cor[0][0] * cor[1][1], cor[0][1] * cor[1][1], cor[1][1]])
    d=d+2
u,s,v=np.linalg.svd(A)
s=0
H=np.reshape(v[8, :], (3, 3))
newImage= 255 * np.ones((1200, 1400, 3))
n = img1.shape[0]
m = img1.shape[1]
for x in range(n):
    for y in range(m):
        newImage[x + 200, y + 200, :]= img2[x, y, :]
        xi=np.array([y,x,1])
        xf=np.matmul(H, xi)
        newImage[int(xf[1] / xf[2]) + 200, int(xf[0] / xf[2]) + 200, :] = img1[x, y, :]
newImage=newImage.astype('uint8')
cv.imshow('Stitched Image', newImage)
cv.waitKey(0)
cv.destroyAllWindows()
