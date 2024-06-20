import cv2 as cv
import numpy as np

def get_3d_points(p1, p2):
    print(p1.shape, p2.shape)
    stereo_calibration = np.load("stereo_calibration.npz")
    intrinsics_1 = np.load("intrinsics/camera_calibration-0.npz")
    intrinsics_2 = np.load("intrinsics/camera_calibration-1.npz")
    K1 = intrinsics_1["calibration_mtx"]
    K2 = intrinsics_2["calibration_mtx"]
    R, T = stereo_calibration['R'], stereo_calibration['T']
    # projects from camera 2 to camera 1
    extrinsic = np.hstack([R, T])
    projection_matrix_1 = np.dot(K1, np.hstack([np.eye(3), np.zeros((3, 1))]))
    projection_matrix_2 = np.dot(K2, extrinsic)
    
    # points4d = cv.triangulatePoints(projection_matrix_1, projection_matrix_2, p1, p2)
    pcd = [triangulate(projection_matrix_1, projection_matrix_2, p1[:, i], p2[:, i]) for i in range(p1.shape[-1])]
    if len(pcd) == 2:
        print(np.linalg.norm(pcd[0] - pcd[1]))
    return np.array(pcd)

def opencv_triangulate(P1, P2, p1, p2):
    points4d = cv.triangulatePoints(P1, P2, p1, p2)
    print('Triangulated point: ', points4d[:3, :] / points4d[3, :])
    return points4d[:3, :] / points4d[3, :]

def triangulate(P1, P2, point1, point2):
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

        print('Triangulated point: ')
        print(Vh[3,0:3]/Vh[3,3])
        return Vh[3,0:3]/Vh[3,3]


def reproject3d():
    cv.projectPoints()