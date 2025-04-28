import numpy as np
import cv2
import glob
import time

path = "output/run4"

rgbImgs = sorted(glob.glob(path + "/rgb_*.npy"))
depthImgs = sorted(glob.glob(path + "/depth_*.npy"))

# print(rgbImgs.__len__())
# print(np.load(rgbImgs[0]).shape)
# print(np.load(depthImgs[0]).shape)

for i in range(rgbImgs.__len__()):
    # print(i)
    rgb = np.load(rgbImgs[i])
    depth = np.load(depthImgs[i])
    maxi = np.max(depth)
    depth = np.where(depth == 0, maxi, depth)
    cv2.imshow("rgb", rgb)
    cv2.imshow("depth", depth)
    if cv2.waitKey(1) == ord('q'):
        break
    time.sleep(.05)

with open(path + '\\imu.npy', 'rb') as f:
    while(1):
        try:
            a = np.load(f)
            print(a)
        except EOFError:
            break

# with open('output\\run1\\rgb_0002041.npy', 'rb') as depth:
#     a = np.load(depth)
#     cv2.imshow("depth", a)
#     while 1:
#         if cv2.waitKey(1) == ord('q'):
#             break