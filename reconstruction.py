import cv2
import numpy as np

import utils as h
import maths as mth

from scipy import optimize as opt


def compute_proj_camera(F, cam_id):
    # Result 9.15 of MVG (v = 0, lambda = 1). It assumes P1 = [I|0]

    ...

    return P


# Write here the method for DLT triangulation
def triangulate(x1, x2, P1, P2, imsize) -> np.ndarray:
    assert P1.shape == (3,4) == P2.shape
    assert x1.shape == x2.shape and x1.shape[0] == 3
    
    #Normalize points to have [x, y, 1]
    x1_scaled = x1 / x1[2,:]
    x2_scaled = x2 / x2[2,:]

    #Use the normalized method to ensure that pixels for both images and coordinates have the same range [-1,1]
    H = np.array([[2/imsize[0], 0,           -1],
                  [0,           2/imsize[1], -1],
                  [0,           0,            1]])
    
    x1_norm = H @ x1_scaled
    x2_norm = H @ x2_scaled
    P1_norm = H @ P1
    P2_norm = H @ P2

    #Compute matrix A independently for each point
    #We want to minimize the error for correspondences on their own, not general error
    X = np.ones((4, x1.shape[1]))
    for ind, point in enumerate(x1_norm.T):
      A = np.array([point[0] * P1_norm[2,:] - P1_norm[0,:],
                    point[1] * P1_norm[2,:] - P1_norm[1,:],
                    x2_norm[0,ind] * P2_norm[2,:] - P2_norm[0,:],
                    x2_norm[1,ind] * P2_norm[2,:] - P2_norm[1,:]])
    
      #We want to find min ||AX||2 -> Least sqaure solved by SVD decomposition
      U,D,V_t = np.linalg.svd(A)
      #Solution is the eigenvector with the smallest eigenvalue
      X[:, ind] = V_t[-1]

    X = X / X[3,:]
    return X

def estimate_3d_points_2(P1, P2, xr1, xr2):
    """
    Linear triangulation (Hartley ch 12.2 pg 312) to find the 3D point X
    where p1 = m1 * X and p2 = m2 * X. Solve AX = 0.
    :param p1, p2: 2D points in homo. or catesian coordinates. Shape (3 x n)
    :param m1, m2: Camera matrices associated with p1 and p2. Shape (3 x 4)
    :returns: 4 x n homogenous 3d triangulated points
    """
    num_points = xr1.shape[1]
    res = np.ones((4, num_points))

    for i in range(num_points):
        A = np.asarray([
            (xr1[0, i] * P1[2, :] - P1[0, :]),
            (xr1[1, i] * P1[2, :] - P1[1, :]),
            (xr2[0, i] * P2[2, :] - P2[0, :]),
            (xr2[1, i] * P2[2, :] - P2[1, :])
        ])

        _, _, V = np.linalg.svd(A)

        X = V[-1, :4]
        res[:, i] = X / X[3]

    return res


def compute_reproj_error(X, P1, P2, xr1, xr2):
    # project 3D points using P

    ...

    return error


def transform(aff_hom, Xprj, cams_pr):
    # Algorithm 19.2 of MVG
    Xaff = np.linalg.inv(aff_hom)
    Xaff = Xaff / Xaff[-1]

    cams_aff = [p@aff_hom for p in cams_pr]

    return Xaff, cams_aff


def homog(x):
    return np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)


def euclid(x):
    return x[:, :-1] / x[:, [-1]]


def compute_eucl_cam(F, x1, x2):
    K = np.array([[2362.12, 0, 1520.69], [0, 2366.12, 1006.81], [0, 0, 1]])
    E = K.T @ F @ K

    # camera projection matrix for the first camera
    P1 = K @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

    # make sure E is rank 2
    U, S, V = np.linalg.svd(E)
    if np.linalg.det(np.dot(U, V)) < 0:
        V = -V
    E = np.dot(U, np.dot(np.diag([1, 1, 0]), V))

    # create matrices (Hartley p 258)
    Z = mth.skew([0, 0, -1])
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    # return all four solutions
    P2 = [np.vstack((np.dot(U, np.dot(W, V)).T, U[:, 2])).T,
          np.vstack((np.dot(U, np.dot(W, V)).T, -U[:, 2])).T,
          np.vstack((np.dot(U, np.dot(W.T, V)).T, U[:, 2])).T,
          np.vstack((np.dot(U, np.dot(W.T, V)).T, -U[:, 2])).T]

    ind = 0
    maxres = 0

    for i in range(4):
        # triangulate inliers and compute depth for each camera
        homog_3D = cv2.triangulatePoints(P1, P2[i], x1[:2], x2[:2])
        # the sign of the depth is the 3rd value of the image point after projecting back to the image
        d1 = np.dot(P1, homog_3D)[2]
        d2 = np.dot(P2[i], homog_3D)[2]

        if sum(d1 > 0) + sum(d2 < 0) > maxres:
            maxres = sum(d1 > 0) + sum(d2 < 0)
            ind = i
            infront = (d1 > 0) & (d2 < 0)

    list_cams = []
    list_cams.append(P1)
    list_cams.append(P2[ind])

    return list_cams
