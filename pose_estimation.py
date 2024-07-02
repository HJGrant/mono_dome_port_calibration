import numpy as np
import cv2
import glob

def draw(img, corners, imgpts):
    corners = corners.astype('int16')
    imgpts = imgpts.astype('int16')

    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)

    return img

def detect_checkerboard_corners(image, pattern_size):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    
    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                   criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        
        with_corners = cv2.drawChessboardCorners(gray, pattern_size, corners2, ret)
        print(corners2)
        cv2.imshow('with_corners', with_corners)
        cv2.waitKey(0)

    return ret, corners2

fs = cv2.FileStorage("monoParams.yml", cv2.FILE_STORAGE_READ)
mtx = fs.getNode("cameraMatrix").mat()
dist = fs.getNode("distCoeffs").mat()
img = cv2.imread("frame.png")
shape_size = (3,6)

objp = np.zeros((shape_size[0]*shape_size[1],3), np.float32)
objp[:,:2] = np.mgrid[0:shape_size[0],0:shape_size[1]].T.reshape(-1,2)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

ret, corners = detect_checkerboard_corners(img, pattern_size=shape_size)

# Find the rotation and translation vectors.
ret,rvecs, tvecs = cv2.solvePnP(objp, corners, mtx, dist)

# project 3D points to image plane
imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

img = draw(img, corners, imgpts)

cv2.imshow('img',img)
k = cv2.waitKey(0) & 0xFF

if k == ord('s'):
    cv2.imwrite('pose_estimation_result.png', img)
    cv2.destroyAllWindows()