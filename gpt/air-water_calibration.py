import numpy as np
import cv2

####################################################
#############DON'T USE THIS CODE####################
####################################################


# Define a function to compute the homography matrix using Gauss-Newton optimization
def compute_homography_GaussNewton(pts_src, pts_dst, num_iterations=100, tolerance=1e-5):
    """
    Compute the homography matrix H that maps points from pts_src to pts_dst
    using Gauss-Newton optimization.

    Args:
    - pts_src: Array of shape (N, 2) containing source points.
    - pts_dst: Array of shape (N, 2) containing destination points.
    - num_iterations: Maximum number of iterations for optimization.
    - tolerance: Convergence tolerance based on change in parameters.

    Returns:
    - H: 3x3 homography matrix.
    """

    num_points = pts_src.shape[0]
    A = np.zeros((2 * num_points, 9))

    # Build the A matrix
    for i in range(num_points):
        x, y = pts_src[i, 0], pts_src[i, 1]
        u, v = pts_dst[i, 0], pts_dst[i, 1]
        A[2*i, :] = [-x, -y, -1, 0, 0, 0, u*x, u*y, u]
        A[2*i+1, :] = [0, 0, 0, -x, -y, -1, v*x, v*y, v]

    # Initial estimate of the homography parameters (flattened into a vector)
    H = np.eye(3)
    H = H.flatten()

    for iteration in range(num_iterations):
        # Compute current projections
        pts_src_proj = np.dot(pts_src, H.reshape(3,3).T)
        pts_src_proj /= pts_src_proj[:, 2][:, np.newaxis]

        # Compute error (difference between actual and projected points)
        error = np.concatenate((pts_dst[:, 0] - pts_src_proj[:, 0],
                                pts_dst[:, 1] - pts_src_proj[:, 1]))

        # Compute Jacobian
        J = np.zeros((2 * num_points, 9))
        for i in range(num_points):
            x, y, z = pts_src[i, 0], pts_src[i, 1], 1
            u, v = pts_dst[i, 0], pts_dst[i, 1]
            J[2*i, :] = [-x/z, -y/z, -1/z, 0, 0, 0, u*x/z, u*y/z, u/z]
            J[2*i+1, :] = [0, 0, 0, -x/z, -y/z, -1/z, v*x/z, v*y/z, v/z]

        # Compute Gauss-Newton update
        JtJ = np.dot(J.T, J)
        Jte = np.dot(J.T, error)
        delta = np.linalg.solve(JtJ, -Jte)

        # Update H
        H += delta

        # Reshape H into a 3x3 matrix
        H_mat = H.reshape(3, 3)

        # Check convergence
        if np.linalg.norm(delta) < tolerance:
            break

    return H_mat

# Example usage:
if __name__ == '__main__':
    # Load images
    img_before = cv2.imread('air_frame.png')
    img_after = cv2.imread('air_frame.png')

    # Find checkerboard corners in both images
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    gray_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY)
    ret_before, corners_before = cv2.findChessboardCorners(gray_before, (3, 6), flags)
    ret_after, corners_after = cv2.findChessboardCorners(gray_after, (3, 6), flags)

    if ret_before and ret_after:
        # Refine corner positions
        corners_before = cv2.cornerSubPix(gray_before, corners_before, (11, 11), (-1, -1), criteria)
        corners_after = cv2.cornerSubPix(gray_after, corners_after, (11, 11), (-1, -1), criteria)

        # Compute homography using Gauss-Newton optimization
        H = compute_homography_GaussNewton(corners_before.squeeze(), corners_after.squeeze())

        # Use H to transform points or warp images
        img_warped = cv2.warpPerspective(img_before, H, (img_after.shape[1], img_after.shape[0]))

        # Display results (optional)
        cv2.imshow('Warped Image', img_warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("Failed to find chessboard corners in both images.")
