from .core import CVCameraOptions
from .mixins import StereoRectificationMapLoader, PairedImageLoader

class DepthEsitmationOptions(StereoRectificationMapLoader, PairedImageLoader):
    def __init__(self, count:int=5, extrinsics_dir:str="", paired_images_path:str="", left_cam_id:int=0, right_cam_id:int=1, cv_options:CVCameraOptions=None) -> None:
        self.count = count
        self.extrinsics_dir = extrinsics_dir
        self.paired_images_path = paired_images_path
        self.left_cam_id = left_cam_id
        self.right_cam_id = right_cam_id
        self.cv_options = cv_options
