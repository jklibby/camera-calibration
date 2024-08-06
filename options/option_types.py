from typing import *

class CVOptionType(TypedDict):
    resolution: Tuple[int, int]
    window_size:Tuple[int, int]

class CameraCaptureType(TypedDict):
    left_cam_id: int
    right_cam_id: int
    image_count: int
    stereo_images_count: int
    stereo_images_path: str
    image_path: str

class CameraCalibrationType(TypedDict):
    pattern_size:Tuple[int, int]
    intrinsic_params_dir:str
    extrinsic_params_dir: str
    headless: bool
    single_calibration_error_threshold: float | None
    stereo_calibration_error_threshold: float | None
    world_scaling: float

class Validation(TypedDict):
    pattern_size:Tuple[int, int]

