from typing import *

class CVOptionType(TypedDict):
    """
        Typed Dict for loading OpenCV options from yaml config file.
    """
    resolution: Tuple[int, int]
    window_size:Tuple[int, int]

class CameraCaptureType(TypedDict):
    """
        Typed Dict for loading Camera capture options 
        from yaml config file for single and stereo camera capture
    """
    left_cam_id: int
    right_cam_id: int
    image_count: int
    stereo_images_count: int
    stereo_images_path: str
    image_path: str

class CameraCalibrationType(TypedDict):
    """
        Typed Dict for loading Camera calibrate options 
        from yaml config file for single and stereo camera calibration
    """
    pattern_size:Tuple[int, int]
    intrinsic_params_dir:str
    extrinsic_params_dir: str
    headless: bool
    single_calibration_error_threshold: float | None
    stereo_calibration_error_threshold: float | None
    world_scaling: float

class Validation(TypedDict):
    """
        Typed Dict for loading Calibration Validation pattern option 
        from yaml config. 
    """
    pattern_size:Tuple[int, int]

