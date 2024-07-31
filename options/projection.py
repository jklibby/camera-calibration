from typing import *

from .mixins import PairedImageLoader, StereoRectificationMapLoader
from .core import CVCameraOptions

class CheckboardProjectionOptions(PairedImageLoader, StereoRectificationMapLoader):
    def __init__(self, left_cam_id: int, right_cam_id:int, pattern_size:Tuple[int, int],
                 validation_pattern_size: Tuple[int, int],
                intrinsics_dir:str, extrinsics_dir:str, world_scaling:float=0, 
                count:int=0, paired_images_path: str| None = None, cv_options: CVCameraOptions|None=None) -> None:
        self.left_cam_id = left_cam_id
        self.right_cam_id = right_cam_id
        self.pattern_size = pattern_size
        self.extrinsics_dir = extrinsics_dir
        self.intrinsic_dir = intrinsics_dir
        self.world_scaling = world_scaling
        self.count = count
        self.validation_pattern_size = validation_pattern_size
        self.paired_images_path = paired_images_path
        self.cv_options = cv_options
