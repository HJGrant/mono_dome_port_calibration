import numpy as np
import cv2

##This code was written by ChatGPT

def detect_checkerboard_corners(image, pattern_size=(3, 6)):
    """
    Detect checkerboard corners in the given image.
    :param image: Input image.
    :param pattern_size: Size of the checkerboard pattern (number of corners per row and column).
    :return: Coordinates of the detected corners.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if ret:
        print("I found corners")
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                   criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        
    cv2.imshow()

    return ret, corners

def gauss_newton_3d(original_points, transformed_points, max_iterations=100, tolerance=1e-6):
    """
    Estimate the transformation matrix using the Gauss-Newton algorithm.
    :param original_points: Original 3D points.
    :param transformed_points: Transformed 3D points.
    :param max_iterations: Maximum number of iterations for the Gauss-Newton algorithm.
    :param tolerance: Convergence tolerance.
    :return: Estimated transformation matrix (3x4).
    """
    # Initialize transformation parameters (12 parameters for 3x4 affine matrix)
    A = np.zeros(12)
    
    def residuals(A, P, Q):
        A_matrix = A.reshape(3, 4)
        P_homogeneous = np.hstack((P, np.ones((P.shape[0], 1))))
        Q_predicted = P_homogeneous @ A_matrix.T
        return (Q_predicted - Q).flatten()
    
    def jacobian(A, P):
        P_homogeneous = np.hstack((P, np.ones((P.shape[0], 1))))
        J = np.zeros((P.shape[0] * 3, 12))
        for i in range(3):
            for j in range(4):
                J[i::3, i * 4 + j] = P_homogeneous[:, j]
        return J
    
    for iteration in range(max_iterations):
        r = residuals(A, original_points, transformed_points)
        J = jacobian(A, original_points)
        delta_A, _, _, _ = np.linalg.lstsq(J, -r, rcond=None)
        A += delta_A
        if np.linalg.norm(delta_A) < tolerance:
            break
    
    return A.reshape(3, 4)

def estimate_transformation_matrix(image_pairs, pattern_size=(9, 6)):
    """
    Estimate the transformation matrix from image pairs.
    :param image_pairs: List of tuples containing image before and after the transformation.
    :param pattern_size: Size of the checkerboard pattern (number of corners per row and column).
    :return: Estimated transformation matrix (3x4).
    """
    original_points = []
    transformed_points = []

    for (img_before, img_after) in image_pairs:
        ret_before, corners_before = detect_checkerboard_corners(img_before, pattern_size)
        ret_after, corners_after = detect_checkerboard_corners(img_after, pattern_size)

        if ret_before and ret_after:
            original_points.extend(corners_before.reshape(-1, 3))
            transformed_points.extend(corners_after.reshape(-1, 3))

    original_points = np.array(original_points)
    transformed_points = np.array(transformed_points)

    transformation_matrix = gauss_newton_3d(original_points, transformed_points)
    return transformation_matrix

# Example usage with image pairs
image_pairs = [
    (cv2.imread('air_water_1.png'), cv2.imread('air_water_2.png'))
]

transformation_matrix = estimate_transformation_matrix(image_pairs)
print("Estimated transformation matrix:")
print(transformation_matrix)