import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img1=cv.cvtColor(cv.imread("C:/Users/Bumuthu Dilshan/Desktop/Assignment 2/img5.ppm"),cv.COLOR_BGR2RGB)
img2=cv.cvtColor(cv.imread("C:/Users/Bumuthu Dilshan/Desktop/Assignment 2/img1.ppm"),cv.COLOR_RGB2BGR)

src = [[382,233],[452,276],[262,378],[473,95]]
dst = [[333,190],[526,222],[30,343],[634,2]]

# generating A
a = np.zeros((8, 9))
for i in range(0,8,2):
    a[i]=np.array([0,0,0,src[int(i/2)][0],src[int(i/2)][1],1,-dst[int(i/2)][1]*src[int(i/2)][0], -dst[int(i/2)][1]*src[int(i/2)][1],-dst[int(i/2)][1]])
    a[i+1]=np.array([src[int(i / 2)][0], src[int(i / 2)][1],1,0,0,0,-dst[int(i / 2)][0] * src[int(i / 2)][0],-dst[int(i / 2)][0] * src[int(i / 2)][1], -dst[int(i / 2)][0]])
# print(a)
u, s, v = np.linalg.svd(a)
h = np.reshape(v[8], (3, 3))

print(h)
n = img1.shape[0]
m = img1.shape[1]
new=255*np.ones((n+500,m+500,3))

for i in range(n+500):
    for j in range(m+500):
        z = np.array([i,j,1])
        x1,y1,alpha = np.matmul(h,z)


       






new = new.astype('uint8')
print(new)
print(n,m)
cv.imshow('dcscio',new)
cv.waitKey(0)
cv.destroyAllWindows()
f, axarr = plt.subplots(1, 2)

axarr[0].set_title("First Image")
axarr[0].imshow(img1)
axarr[1].set_title('Second Image')
axarr[1].imshow(new)
plt.show()



