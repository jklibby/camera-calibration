import cv2 as cv
import numpy as np

def get_3d_points(p1, p2, dir):
    print(p1.shape, p2.shape)
    stereo_calibration = np.load("extrinsics/{}/stereo_calibration.npz".format(dir))
    intrinsics_1 = np.load("intrinsics/{}/camera_calibration_0.npz".format(dir))
    intrinsics_2 = np.load("intrinsics/{}/camera_calibration_1.npz".format(dir))
    K1 = intrinsics_1["calibration_mtx"]
    K2 = intrinsics_2["calibration_mtx"]
    R, T = stereo_calibration['R'], stereo_calibration['T']
    print(T)
    # projects from camera 2 to camera 1
    extrinsic = np.hstack([R, T])
    projection_matrix_1 = np.dot(K1, np.hstack([np.eye(3), np.zeros((3, 1))]))
    projection_matrix_2 = np.dot(K2, extrinsic)
    
    # points4d = cv.triangulatePoints(projection_matrix_1, projection_matrix_2, p1, p2)
    pcd = [triangulate(projection_matrix_1, projection_matrix_2, p1[i, :], p2[i, :]) for i in range(p1.shape[0])]
    if len(pcd) == 1:
        print(np.linalg.norm(pcd[0]))
    if len(pcd) == 2:
        # TO-DO: Get world scaling from config
        print(np.linalg.norm(pcd[0] - pcd[1]) * 1.5)
    return np.array(pcd)

def opencv_triangulate(P1, P2, p1, p2):
    points4d = cv.triangulatePoints(P1, P2, p1, p2)
    print('Triangulated point: ', points4d[:3, :] / points4d[3, :])
    return points4d[:3, :] / points4d[3, :]

def triangulate(P1, P2, point1, point2):
    point1 = point1.reshape(-1)
    point2 = point2.reshape(-1)
    A = np.array([point1[1]*P1[2,:] - P1[1,:],
        P1[0,:] - point1[0]*P1[2,:],
        point2[1]*P2[2,:] - P2[1,:],
        P2[0,:] - point2[0]*P2[2,:]
    ])
    A = A.reshape((4,4))
    #print('A: ')
    #print(A)

    B = A.T @ A
    from scipy import linalg
    U, s, Vh = linalg.svd(B, full_matrices = False)

    return Vh[3,0:3]/Vh[3,3]


def reproject3d():
    cv.projectPoints()