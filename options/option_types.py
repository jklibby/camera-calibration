from typing import *

class CVOptionType(TypedDict):
    resolution: Tuple[int, int]
    window_size:Tuple[int, int]

class SingleCameraCaptureType(TypedDict):
    cam_id: int
    image_count: int
    image_path: str

class SingleCameraCalibrationType(TypedDict):
    cam_id:int
    count:int
    image_path:str 
    pattern_size:Tuple[int, int]
    flags:int | None
    criteria:int|None
    intrinsic_params_dir:str
    headless: bool 
    error_threshold: float | None     

class StereoCameraCaptureType(TypedDict):
    left_cam_id: int
    right_cam_id: int
    count:int
    path: str 

class StereoCameraCalibrationType(TypedDict):
    left_cam_id:int
    right_cam_id:int
    count:int
    image_path:int
    pattern_size:Tuple[int, int]
    flags:int | None
    criteria:int|None
    intrinsic_dir:str
    extrinsics_dir:str
    headless:bool
    error_threshold:float

class StereoRectificationType(TypedDict):
    count:int
    left_cam_id:int
    right_cam_id:int
    image_path:str
    dir:str
    intrinsic_dir:str
    extrinsic_dir:str
    flags:int|None
    headless:bool

class CheckboardProjectionType(TypedDict):
    left_cam: int
    right_cam:int
    pattern_size:Tuple[int, int]
    validation_pattern_size:Tuple[int, int]
    intrinsics_dir:str
    extrinsics_dir:str
    world_scaling:float
    count:int
    paired_images_path: str| None