from .core import CVCameraOptions
from .mixins import StereoRectificationMapLoader, PairedImageLoader

class DepthEstimationOptions(StereoRectificationMapLoader, PairedImageLoader):
    """
    A class for configuring options related to tuning depth estimation using Block Matching and Semi Global Block Matching.

    Attributes:
        count (int): The number of image pairs to be used for depth estimation.
        extrinsics_dir (str): The directory containing the extrinsic parameters.
        paired_images_path (str): The path to the directory containing paired stereo images.
        left_cam_id (int): The ID of the left camera.
        right_cam_id (int): The ID of the right camera.
        cv_options (CVCameraOptions): Camera options for OpenCV.
    """
    def __init__(self, count:int=5, extrinsics_dir:str="", paired_images_path:str="", left_cam_id:int=0, right_cam_id:int=1, cv_options:CVCameraOptions=None) -> None:
        self.count = count
        self.extrinsics_dir = extrinsics_dir
        self.paired_images_path = paired_images_path
        self.left_cam_id = left_cam_id
        self.right_cam_id = right_cam_id
        self.cv_options = cv_options
