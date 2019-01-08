import matplotlib.pyplot as plt
import cv2
import numpy as np
def getPanaroma(img1, img2, h, l, dx, dy):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    resultImg = 255 * np.ones((h, l, 3))
    for x in range(img1.shape[0]):
        for y in range(img1.shape[1]):
            resultImg[x + dx, y + dy, :] = img2[x, y, :]
            xi = np.array([y, x, 1])
            xf = np.matmul(M, xi)
            resultImg[int(xf[1] / xf[2]) + dx, int(xf[0] / xf[2] + dy), :] = img1[x, y, :]
    resultImg = resultImg.astype('uint8')
    return resultImg
img1 = cv2.cvtColor(cv2.imread('C:/Users/Bumuthu Dilshan/Desktop/Assignment 2/img2.ppm', cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(cv2.imread('C:/Users/Bumuthu Dilshan/Desktop/Assignment 2/img5.ppm', cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
img3 = cv2.cvtColor(cv2.imread('C:/Users/Bumuthu Dilshan/Desktop/Assignment 2/img3.ppm', cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
imidiateImg = getPanaroma(img1, img2, 940, 800, 160, 0)
finalImg = getPanaroma(img3, imidiateImg, 1050, 800, 20, 0)
plt.imshow(finalImg)
plt.show()
