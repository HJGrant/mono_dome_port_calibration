import cv2
import numpy as np

img_file = "./frame.png"
img = cv2.imread(img_file)
fs = cv2.FileStorage("monoParams.yml", cv2.FILE_STORAGE_READ)


mtx = fs.getNode("cameraMatrix").mat()
dist = fs.getNode("distCoeffs").mat()

#get the optimal camera matrix
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
 
#crop the image
x, y, w, h = roi
dst_cropped = dst[y:y+h, x:x+w]
cv2.imwrite('mono_calib_result.png', dst)

print((dst==img).all())

img = cv2.resize(img, (960, 480))
dst = cv2.resize(dst, (960, 480))

stacked = np.hstack([img, dst])
cv2.imshow('STACKED', stacked)
cv2.waitKey(0)