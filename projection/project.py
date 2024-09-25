from typing import * 
import numpy as np
import cv2 as cv
from pathlib import Path

from utils import get_CB_corners

from options import CheckboardProjectionOptions

def point_cloud_selector(opts: CheckboardProjectionOptions) -> np.ndarray:
    """
    Captures points selected by the users as pixel co-ordinates in stereo pair of images. 
    Projects those points from 2D to 3D using DLT. Then renders these points in a Visualizer.

    Args:
        opts (CheckboardProjectionOptions): Configuration options for checkerboard projection and stereo camera setup.

    Returns:
        pcd: np.ndarray: 3D point cloud of selected points in stereo pair of images.
    """
    path = Path(opts.extrinsics_dir).joinpath(*["stereo_rectification", "validation_images"])
    left_cap = FrameReader(path, opts.left_cam_id)
    right_cap = FrameReader(path, opts.right_cam_id)
    cv.namedWindow("Left Rectified Image")
    cv.namedWindow("Right Rectified Image")
    selected_frames = list()
    while True:
        lret, left_frame = next(left_cap.read())
        rret, right_frame = next(right_cap.read())
        
        if not (lret or rret):
            print("Cannot read video frames from: ", str(path))
            exit()
        left_frame_rect, right_frame_rect = _write_to_frames(left_frame, right_frame, "Press space to select frames. Selected Frames: {}".format(len(selected_frames)))
        break_loop = False
        while True:
            cv.imshow("Left Rectified Image", left_frame_rect)
            cv.imshow("Right Rectified Image", right_frame_rect)
            key = cv.waitKey(1) & 0xFF

            if key == ord(' '):
                selected_frames.append((left_frame, right_frame))
                break
            
            if key == ord('q') or len(selected_frames) >= 5:
                break_loop = True
                break
        
        if break_loop:
            break

    cv.destroyAllWindows()

    idx = 0
    left_point_dropper = PointDropper(img=selected_frames[idx][0])
    right_point_dropper = PointDropper(img=selected_frames[idx][1])
    cv.namedWindow("Left Point Selector")
    cv.namedWindow("Right Point Selector")
    cv.setMouseCallback("Left Point Selector", left_point_dropper.on_mouse)
    cv.setMouseCallback("Right Point Selector", right_point_dropper.on_mouse)

    left_points = list()
    right_points = list()
    while True:
        cv.imshow("Left Point Selector", left_point_dropper.img)
        cv.imshow("Right Point Selector", right_point_dropper.img)
        key = cv.waitKey(1) & 0xFF

        if key == ord(' '):
            # flush selected points
            if left_point_dropper.points and right_point_dropper.points and len(left_point_dropper.points) == len(right_point_dropper.points):
                left_points.append(left_point_dropper.points)
                right_points.append(right_point_dropper.points)
                cv.imwrite("grid-points-left.png", left_point_dropper.img)
                cv.imwrite("grid-points-right.png", right_point_dropper.img)
            idx += 1
            if idx >= len(selected_frames):
                cv.destroyAllWindows()
                break
            left_point_dropper = PointDropper(img=selected_frames[idx][0])
            right_point_dropper = PointDropper(img=selected_frames[idx][1])
            cv.setMouseCallback("Left Point Selector", left_point_dropper.on_mouse)
            cv.setMouseCallback("Right Point Selector", right_point_dropper.on_mouse)
    
    # get 3D projection
    left_points = np.array(left_points)
    right_points = np.array(right_points)
    num_images = left_points.shape[0]
    print(left_points.shape, right_points.shape)
    left_points = left_points.reshape((-1, 1, 2))
    right_points = right_points.reshape((-1, 1, 2))
    pcd = get_point_cloud(left_points, right_points, opts)
    pcd = pcd.reshape((num_images, -1, 3))
    print(pcd.shape, np.linalg.norm(pcd[:, 0, :] - pcd[:, -1, :]))
    return pcd


def measure_checkerboard(opts: CheckboardProjectionOptions)-> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """
    Measures the checkerboard pattern in a stereo camera setup by capturing rectified images from both cameras.

    Args:
        opts (CheckboardProjectionOptions): Configuration options for checkerboard projection and stereo camera setup.

    Returns:
        Tuple[np.ndarray, List[Tuple[float, float]]]: 
            - A numpy array of point clouds corresponding to the checkerboard corners detected across multiple frames.
            - A list of tuples where each tuple contains the width and height of the checkerboard pattern detected in each frame.
    """
    left_map_x, left_map_y, right_map_x, right_map_y = opts.load_remaps()

    left_cap = cv.VideoCapture(opts.left_cam_id)
    right_cap = cv.VideoCapture(opts.right_cam_id)
    video_err = not (left_cap.isOpened() and right_cap.isOpened())
    if video_err:
        print("Cannot read video frames {}".format(video_err))
        exit()
    opts.cv_options.named_window("Left Rectified Image")
    opts.cv_options.named_window("Right Rectified Image")
    pattern = opts.validation_pattern_size
    pcd, height, width = np.array([]), -1, -1
    dimensions = list()
    pcd_list = list()
    while True:
        lret, left_frame = left_cap.read()
        rret, right_frame = right_cap.read()
        left_frame_rect = cv.remap(left_frame, left_map_x, left_map_y, cv.INTER_LANCZOS4)
        right_frame_rect = cv.remap(right_frame, right_map_x, right_map_y, cv.INTER_LANCZOS4)
        cv.imshow("Left Rectified Image", left_frame_rect)
        cv.imshow("Right Rectified Image", right_frame_rect)
        key = cv.waitKey(1) & 0xFF

        if key == ord(' '):
            lret_corners, _, lcorners = get_CB_corners(left_frame_rect, pattern)
            rret_corners, _, rcorners = get_CB_corners(right_frame_rect, pattern)
            if lret_corners and rret_corners:
                pcd = get_point_cloud(lcorners, rcorners, opts)
                width, height = _get_width_height(pcd, pattern)
                print(width, height)
                dimensions.append((width, height))
                pcd_list.append(pcd)
                

        if key == ord('q'):
            left_cap.release()
            right_cap.release()
            cv.destroyAllWindows()
            break
        
        if not (rret and lret):
            print("Cannot read video frames")
            break
    
    left_cap.release()
    right_cap.release()
    cv.destroyAllWindows()
    return np.array(pcd_list), dimensions

def get_checkerboard_pcd(opts: CheckboardProjectionOptions) -> np.ndarray:
    """
    Computes the point cloud of the checkerboard pattern from a set of stereo images. 
    The function loads paired stereo images, applies rectification, and detects checkerboard corners.
    It then triangulates the detected corners to compute the 3D point cloud.

    Args:
        opts (CheckboardProjectionOptions): Configuration options for checkerboard projection and stereo camera setup.

    Returns:
        np.ndarray: A numpy array of point clouds corresponding to the checkerboard corners detected across multiple images.
    """
    left_map_x, left_map_y, right_map_x, right_map_y = opts.load_remaps()
    
    pattern = opts.pattern_size
    # load paired images
    images = opts.load_paired_images()
    corners = list()
    for li, ri in images:
        left_frame = cv.imread(li)
        right_frame = cv.imread(ri)

        # apply rectification
        left_frame_rect = cv.remap(left_frame, left_map_x, left_map_y, cv.INTER_LANCZOS4)
        right_frame_rect = cv.remap(right_frame, right_map_x, right_map_y, cv.INTER_LANCZOS4)

        # detect checkerboards
        lret, _, left_corner = get_CB_corners(left_frame_rect, pattern)
        rret, _, right_corner = get_CB_corners(right_frame_rect, pattern)
        if lret and rret:
            corner_pcd = get_point_cloud(left_corner, right_corner, opts)
            width, height = _get_width_height(corner_pcd, pattern)
            print(width, height)
            corners.append(corner_pcd)
    
    return np.array(corners)

def get_point_cloud(p1, p2, opts: CheckboardProjectionOptions) -> np.ndarray:
    """
    Triangulates points from stereo image pairs to generate a 3D point cloud.

    Args:
        p1 (np.ndarray): Detected checkerboard corners in the left image.
        p2 (np.ndarray): Detected checkerboard corners in the right image.
        opts (CheckboardProjectionOptions): Configuration options for checkerboard projection and stereo camera setup.

    Returns:
        np.ndarray: A numpy array representing the 3D point cloud of the detected checkerboard pattern.
    """
    stereo_calibration = np.load(Path(opts.extrinsics_dir).absolute().joinpath("stereo_calibration.npz"))
    intrinsics_1 = np.load(Path(opts.intrinsic_dir).absolute().joinpath("camera_calibration_{}.npz".format(opts.left_cam_id)))
    intrinsics_2 = np.load(Path(opts.intrinsic_dir).absolute().joinpath("camera_calibration_{}.npz".format(opts.right_cam_id)))
    K1 = intrinsics_1["calibration_mtx"]
    K2 = intrinsics_2["calibration_mtx"]
    R, T = stereo_calibration['R'], stereo_calibration['T']
    # projects from camera 2 to camera 1
    extrinsic = np.hstack([R, T])
    projection_matrix_1 = np.dot(K1, np.hstack([np.eye(3), np.zeros((3, 1))]))
    projection_matrix_2 = np.dot(K2, extrinsic)
    
    # points4d = cv.triangulatePoints(projection_matrix_1, projection_matrix_2, p1, p2)
    pcd = [_triangulate(projection_matrix_1, projection_matrix_2, p1[i, :], p2[i, :]) for i in range(p1.shape[0])]
    return np.array(pcd)

def _opencv_triangulate(P1, P2, p1, p2):
    points4d = cv.triangulatePoints(P1, P2, p1, p2)
    print('Triangulated point: ', points4d[:3, :] / points4d[3, :])
    return points4d[:3, :] / points4d[3, :]

def _triangulate(P1, P2, point1, point2):
    point1 = point1.reshape(-1)
    point2 = point2.reshape(-1)
    A = np.array([point1[1]*P1[2,:] - P1[1,:],
        P1[0,:] - point1[0]*P1[2,:],
        point2[1]*P2[2,:] - P2[1,:],
        P2[0,:] - point2[0]*P2[2,:]
    ])
    A = A.reshape((4,4))
    
    B = A.T @ A
    U, s, Vh = np.linalg.svd(B, full_matrices = False)

    return Vh[3,0:3]/Vh[3,3]

def _get_width_height(corner_pcd, pattern):
    print(corner_pcd.shape)
    edge_points = [corner_pcd[0], corner_pcd[pattern[0] - 1], corner_pcd[np.prod(pattern) - pattern[0]]]

    width = np.linalg.norm(edge_points[0] - edge_points[1])
    height = np.linalg.norm(edge_points[0] - edge_points[2])
    return width, height

def _write_to_frames(left:np.ndarray, right:np.ndarray, text:str) -> Tuple[np.ndarray, np.ndarray]:
    set_text = lambda frame, text: cv.putText(frame, text ,(100, 100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3, 1)
    return set_text(left, text), set_text(right, text)


def project_world_to_camera(world_points, rvec, tvec):
    ones = np.ones((world_points.shape[0], 1))
    world_points_homogenous = np.hstack([world_points, ones])
    R_t = np.hstack([cv.Rodrigues(rvec)[0], tvec])
    return R_t.dot(world_points_homogenous.T).T


class PointDropper:
    def __init__(self, img) -> None:
        self.img = img
        self.points = list()
    
    def on_mouse(self, event, x, y, flags, params):
        if event == cv.EVENT_LBUTTONDOWN:
            print('x = %d, y = %d'%(x, y))
            self.points.append((x, y))
            self.img = cv.circle(self.img,(x,y),10,(255,0,0),-1)

class FrameReader:
    def __init__(self, path:Path, cam_id:int):
        self.path = path
        self.cam_id = cam_id
        self.count = 0

    def read(self):
        image_path = self.path.joinpath(*["camera{}_{}.png".format(self.cam_id, self.count)])
        self.count += 1
        if image_path.exists():
            yield True, cv.imread(str(image_path))
        else:
            yield False, None