import cv2 as cv
from typing import *
from pathlib import Path

from .mixins import SingleImageLoader, PairedImageLoader

class CVCameraOptions:
    def __init__(self, resolution: Tuple[int, int], window_size:Tuple[int, int]):
        self.resolution = resolution
        self.window_size = window_size
    
    def named_window(self, window_name:str):
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        cv.resizeWindow(window_name, self.window_size[0], self.window_size[1])
    

class SingleCameraCaptureOptions:
    def __init__(self, cam_id: int, image_count: int, image_path: str, cv_options: CVCameraOptions) -> None:
        self.cam_id = cam_id
        self.count = image_count
        self.path = '{}/{}'.format(image_path, cam_id)

        self.cv_options = cv_options

class StereoCameraCaptureOptions:
    def __init__(self, left_cam_id: int, right_cam_id: int, count:int, path: str, cv_options: CVCameraOptions) -> None:
        self.left_cam_id = left_cam_id
        self.right_cam_id = right_cam_id
        self.count = count
        self.path = str(Path(path).absolute())

        self.cv_options = cv_options

class SingleCameraCalibrateOptions(SingleImageLoader):
    def __init__(self, cam_id:int, count:int, image_path:str, 
        pattern_size:Tuple[int, int], flags:int | None=None, criteria:int|None=None,
        intrinsic_params_dir:str=None, headless=False, 
        error_threshold: float | None=None, cv_options:CVCameraOptions=None) -> None:
        self.cam_id = cam_id
        self.count = count
        self.path = image_path
        self.pattern_size = pattern_size
        self.flags = flags
        self.criteria = criteria
        self.dir = intrinsic_params_dir
        self.headless = headless
        self.error_threshold = error_threshold
        
        self.cv_options = cv_options

class StereoCameraCalibrationOptions(PairedImageLoader):
    def __init__(self, left_cam_id:int, right_cam_id:int, count:int, 
            image_path:int, pattern_size:Tuple[int, int], flags:int=None,
            intrinsic_dir:str=None, extrinsics_dir:str=None, headless:bool=False, criteria: int|None=None,
            error_threshold:float=1, cv_options:CVCameraOptions=None) -> None:
        self.left_cam_id = left_cam_id
        self.right_cam_id = right_cam_id
        self.count = count
        self.paired_images_path = str(Path(image_path).absolute())
        self.pattern_size = pattern_size
        self.flags = flags
        self.criteria = criteria
        self.criteria = criteria
        self.dir =  str(Path(extrinsics_dir).absolute())
        self.intrinsics_dir =  str(Path(intrinsic_dir).absolute())
        self.headless = headless
        self.error_threshold = error_threshold

        self.cv_options = cv_options


class StereoCameraRectificationOptions(PairedImageLoader):
    def __init__(self, count:int, left_cam_id:int, right_cam_id:int, image_path:str, dir:str, intrinsic_dir:str, extrinsic_dir:str, flags:Tuple[int, int]|None = None, headless:bool=False, cv_options:CVCameraOptions=None) -> None:
        self.count = count
        self.left_cam_id = left_cam_id
        self.right_cam_id = right_cam_id
        self.paired_images_path =  str(Path(image_path).absolute())
        self.intrinsic_dir = str(Path(intrinsic_dir).absolute())
        self.extrinsic_dir = str(Path(extrinsic_dir).absolute())
        self.flags = flags
        self.headless = headless

        self.cv_options = cv_options