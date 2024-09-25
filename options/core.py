import cv2 as cv
from typing import *
from pathlib import Path

from .mixins import SingleImageLoader, PairedImageLoader

class CVCameraOptions:
    """
    A class to represent camera options for OpenCV.

    Attributes:
        resolution (Tuple[int, int]): The resolution of the camera.
        window_size (Tuple[int, int]): The size of the window to display the camera feed.
    """
    def __init__(self, resolution: Tuple[int, int], window_size:Tuple[int, int]):
        self.resolution = resolution
        self.window_size = window_size
    
    def named_window(self, window_name:str):
        """
        Creates a named window with the specified size.

        Args:
            window_name (str): The name of the window to be created.
        """
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        cv.resizeWindow(window_name, self.window_size[0], self.window_size[1])
    

class SingleCameraCaptureOptions:
    """
        A class to define options for capturing images with a single camera.

        Args:
            cam_id (int): The camera ID.
            image_count (int): The number of images to capture.
            image_path (str): The directory path to save captured images.
            cv_options (CVCameraOptions): Camera options for OpenCV.
        """
    def __init__(self, cam_id: int, image_count: int, image_path: str, cv_options: CVCameraOptions) -> None:
        self.cam_id = cam_id
        self.count = image_count
        self.path = str(Path('{}/{}'.format(image_path, cam_id)).absolute())

        self.cv_options = cv_options

class StereoCameraCaptureOptions:
    """
    A class to define options for capturing paired images with a stereo camera.

    Attributes:
        left_cam_id (int): The ID of the left camera.
        right_cam_id (int): The ID of the right camera.
        count (int): The number of image pairs to capture.
        path (str): The path where the images will be saved.
        rectify_images_opt (StereoCameraRectificationOptions): Capture rectified pair of stereo images.
        cv_options (CVCameraOptions): Camera options for OpenCV.
    """
    def __init__(self, left_cam_id: int, right_cam_id: int, count:int, path: str, cv_options: CVCameraOptions) -> None:
        self.left_cam_id = left_cam_id
        self.right_cam_id = right_cam_id
        self.count = count
        self.path = str(Path(path).absolute())

        self.cv_options = cv_options

        # in-memory flag. Only to be used when stereo calibration is performed
        self.rectify_opts:StereoCameraRectificationOptions | None = None

class SingleCameraCalibrateOptions(SingleImageLoader):
    """
    A class to define options for single camera calibration.

    Attributes:
        cam_id (int): The camera ID.
        count (int): The number of images to use for calibration.
        path (str): The path to the directory containing calibration images.
        pattern_size (Tuple[int, int]): The size of the calibration pattern.
        flags (int | None): Optional flags for calibration.
        criteria (int | None): Optional termination criteria for the calibration process.
        dir (str): Directory to save intrinsic parameters.
        headless (bool): Whether to run calibration in headless mode.
        error_threshold (float | None): Optional threshold for acceptable calibration error.
        cv_options (CVCameraOptions): Camera options for OpenCV.
    """

    def __init__(self, cam_id:int, count:int, image_path:str, 
        pattern_size:Tuple[int, int], flags:int | None=None, criteria:int|None=None,
        intrinsic_params_dir:str=None, headless=False, 
        error_threshold: float | None=None, cv_options:CVCameraOptions=None) -> None:
        self.cam_id = cam_id
        self.count = count
        self.path = str(Path(image_path).absolute())
        self.pattern_size = pattern_size
        self.flags = flags
        self.criteria = criteria
        self.dir = str(Path(intrinsic_params_dir).absolute())
        self.headless = headless
        self.error_threshold = error_threshold
        
        self.cv_options = cv_options

class StereoCameraCalibrationOptions(PairedImageLoader):
    """
    A class to define options for stereo camera calibration.

    Attributes:
        left_cam_id (int): The ID of the left camera.
        right_cam_id (int): The ID of the right camera.
        count (int): The number of image pairs to use for calibration.
        paired_images_path (str): The path to the directory containing paired calibration images.
        pattern_size (Tuple[int, int]): The size of the calibration pattern.
        flags (int | None): Optional flags for calibration.
        criteria (int | None): Optional termination criteria for the calibration process.
        dir (str): Directory to save extrinsic parameters.
        intrinsics_dir (str): Directory to save intrinsic parameters.
        headless (bool): Whether to run calibration in headless mode.
        error_threshold (float): The threshold for acceptable calibration error.
        cv_options (CVCameraOptions): Camera options for OpenCV.
    """
    def __init__(self, left_cam_id:int, right_cam_id:int, count:int, 
            image_path:int, pattern_size:Tuple[int, int], flags:int=None,
            intrinsic_params_dir:str=None, extrinsic_params_dir:str=None, headless:bool=False, criteria: int|None=None,
            error_threshold:float=1, cv_options:CVCameraOptions=None) -> None:
        self.left_cam_id = left_cam_id
        self.right_cam_id = right_cam_id
        self.count = count
        self.paired_images_path = str(Path(image_path).absolute())
        self.pattern_size = pattern_size
        self.flags = flags
        self.criteria = criteria
        self.criteria = criteria
        self.dir =  str(Path(extrinsic_params_dir).absolute())
        self.intrinsics_dir =  str(Path(intrinsic_params_dir).absolute())
        self.headless = headless
        self.error_threshold = error_threshold

        self.cv_options = cv_options


class StereoCameraRectificationOptions(PairedImageLoader):
    """
    A class to define options for stereo camera rectification.

    Attributes:
        count (int): The number of image pairs to use for rectification.
        left_cam_id (int): The ID of the left camera.
        right_cam_id (int): The ID of the right camera.
        paired_images_path (str): The path to the directory containing paired images.
        intrinsic_dir (str): Directory to save intrinsic parameters.
        extrinsic_dir (str): Directory to save extrinsic parameters.
        flags (Tuple[int, int] | None): Optional flags for rectification.
        headless (bool): Whether to run rectification in headless mode.
        cv_options (CVCameraOptions): Camera options for OpenCV.
    """
    def __init__(self, count:int, left_cam_id:int, right_cam_id:int, image_path:str, intrinsic_params_dir:str, extrinsic_params_dir:str, flags:Tuple[int, int]|None = None, headless:bool=False, cv_options:CVCameraOptions=None) -> None:
        self.count = count
        self.left_cam_id = left_cam_id
        self.right_cam_id = right_cam_id
        self.paired_images_path =  str(Path(image_path).absolute())
        self.intrinsic_dir = str(Path(intrinsic_params_dir).absolute())
        self.extrinsic_dir = str(Path(extrinsic_params_dir).absolute())
        self.flags = flags
        self.headless = headless

        self.cv_options = cv_options