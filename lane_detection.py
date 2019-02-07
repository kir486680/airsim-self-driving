from book import *
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('pic.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst = pipeline(img)
dst = perspective_warp(dst, dst_size=(1280,720))
out_img, curves, lanes, ploty = sliding_window(dst)
print(dst)
#plt.imshow(out_img)
#plt.plot(curves[0], ploty, color='yellow', linewidth=1)
#plt.plot(curves[1], ploty, color='yellow', linewidth=1)
print(np.asarray(curves).shape)
curverad=get_curve(img, curves[0],curves[1])
print(curverad)
img = cv2.imread("pic.jpg")
img_ = draw_lanes(img, curves[0], curves[1])
cv2.imshow("img"  , img_)
cv2.waitKey(20000)