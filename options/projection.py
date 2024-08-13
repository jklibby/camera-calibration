from typing import *

from .mixins import PairedImageLoader, StereoRectificationMapLoader
from .core import CVCameraOptions

class CheckboardProjectionOptions(PairedImageLoader, StereoRectificationMapLoader):
    """
    A class to configure options for projecting a checkerboard pattern in a stereo camera setup.

    Attributes:
        left_cam_id (int): The ID of the left camera.
        right_cam_id (int): The ID of the right camera.
        pattern_size (Tuple[int, int]): The size of the checkerboard pattern used for calibration (rows, cols).
        validation_pattern_size (Tuple[int, int]): The size of the validation checkerboard pattern (rows, cols).
        extrinsics_dir (str): The directory containing the extrinsic parameters.
        intrinsic_dir (str): The directory containing the intrinsic parameters.
        world_scaling (float): Scaling factor for converting pattern units to real-world units.
        count (int): The number of image pairs to be used.
        paired_images_path (str | None): The path to the directory containing paired stereo images.
        cv_options (CVCameraOptions | None): Camera options for OpenCV.
        headless (bool | None): Whether to run the checkerboard projection in headless mode.
    """
    def __init__(self, left_cam_id: int, right_cam_id:int, pattern_size:Tuple[int, int],
                 validation_pattern_size: Tuple[int, int],
                intrinsic_params_dir:str, extrinsic_params_dir:str, world_scaling:float=0, 
                count:int=0, paired_images_path: str| None = None, cv_options: CVCameraOptions|None=None, 
                headless: bool | None = False) -> None:
        self.left_cam_id = left_cam_id
        self.right_cam_id = right_cam_id
        self.pattern_size = pattern_size
        self.extrinsics_dir = extrinsic_params_dir
        self.intrinsic_dir = intrinsic_params_dir
        self.world_scaling = world_scaling
        self.count = count
        self.validation_pattern_size = validation_pattern_size
        self.paired_images_path = paired_images_path
        self.cv_options = cv_options
        self.headless = headless
