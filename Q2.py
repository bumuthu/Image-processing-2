import matplotlib.pyplot as plt
import cv2
import numpy as np

img1 = cv2.cvtColor(cv2.imread('C:/Users/Bumuthu Dilshan/Desktop/Assignment 2/img2.ppm', cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(cv2.imread('C:/Users/Bumuthu Dilshan/Desktop/Assignment 2/img5.ppm', cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
better = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        better.append(m)
src_pts = np.float32([kp1[m.queryIdx].pt for m in better]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in better]).reshape(-1, 1, 2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
resultImg = 255 * np.ones((1000, 800, 3))
for x in range(img1.shape[0]):
    for y in range(img1.shape[1]):
        resultImg[x + 150, y, :] = img2[x, y, :]
        xi = np.array([y, x, 1])
        xf = np.matmul(M, xi)
        resultImg[int(xf[1] / xf[2]) + 150, int(xf[0] / xf[2]), :] = img1[x, y, :]
resultImg = resultImg.astype(np.uint8)
plt.imshow(resultImg)
plt.show()
img3 = cv2.drawMatches(img1, kp1, img2, kp2, better, None, flags=2)
plt.imshow(img3)
plt.show()