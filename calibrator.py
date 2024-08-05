import abc
from typing import *
import os
import yaml
from pathlib import Path
import numpy as np
import cv2 as cv

from core import capture_single_image, capture_paired_images
from core import single_camera_calibrate, stereo_camera_calibrate, stereo_rectification, single_camera_rectification
from options import CVOptionType, SingleCameraCalibrationType, SingleCameraCaptureType
from options import StereoCameraCalibrationType, StereoCameraCaptureType, StereoRectificationType
from options import CVCameraOptions, StereoCameraCalibrationOptions, StereoCameraRectificationOptions
from options import SingleCameraCaptureOptions, SingleCameraCalibrateOptions, StereoCameraCaptureOptions
from options import CheckboardProjectionOptions, CheckboardProjectionType, DepthEsitmationOptions
from projection import measure_checkerboard, get_checkerboard_pcd
from visualization import CalibrationVisualizer
from depth_estimation.stereo_depth import get_stereo_depth, get_live_stereo_depth


class CalibratorConfig(TypedDict):
    cv_options: CVOptionType
    left_camera_capture: SingleCameraCaptureType
    right_camera_capture: SingleCameraCaptureType
    
    left_camera_calibration: SingleCameraCalibrationType
    right_camera_calibration: SingleCameraCalibrationType

    stereo_camera_capture: StereoCameraCaptureType
    stereo_camera_calibration: StereoCameraCalibrationType
    stereo_rectification: StereoRectificationType

    cb_projection: CheckboardProjectionType


def load_yaml(file_path:str) -> Dict:
    # ensure file exists
    if not os.path.exists(file_path):
        raise Exception("File not found")
    
    # load file
    config = dict()
    with open(file_path, "r") as yaml_data:
        config = yaml.load(yaml_data, Loader=yaml.Loader)
    
    # load get yaml dict
    return config


class StereoCalibrator:
    cv_options: CVCameraOptions

    left_camera_capture: SingleCameraCaptureOptions
    right_camera_capture: SingleCameraCalibrateOptions
    stereo_camera_capture: StereoCameraCaptureOptions

    left_camera_calibrate: SingleCameraCalibrateOptions
    right_camera_calibrate: SingleCameraCalibrateOptions
    stereo_camera_calibrate: StereoCameraCalibrationOptions

    stereo_rectification: StereoCameraRectificationOptions
    stereo_depth_estimation: DepthEsitmationOptions

    cb_projection: CheckboardProjectionOptions

    @staticmethod
    def from_yaml(file_path: str) -> 'StereoCalibrator':
        config: CalibratorConfig = load_yaml(file_path)
        return StereoCalibrator.from_config(config)
    
    @staticmethod
    def from_config(config: CalibratorConfig) -> 'StereoCalibrator':
        calibrator = StereoCalibrator()
        calibrator.cv_options = CVCameraOptions(**config["cv_options"])
        calibrator.left_camera_capture = SingleCameraCaptureOptions(**config["left_camera_capture"], cv_options=calibrator.cv_options)
        calibrator.right_camera_capture = SingleCameraCaptureOptions(**config["right_camera_capture"], cv_options=calibrator.cv_options)
        calibrator.left_camera_calibrate = SingleCameraCalibrateOptions(**config["left_camera_calibration"], cv_options=calibrator.cv_options)
        calibrator.right_camera_calibrate = SingleCameraCalibrateOptions(**config["right_camera_calibration"], cv_options=calibrator.cv_options)
        
        calibrator.stereo_camera_capture = StereoCameraCaptureOptions(**config["stereo_camera_capture"], cv_options=calibrator.cv_options)
        calibrator.stereo_camera_calibrate = StereoCameraCalibrationOptions(**config["stereo_camera_calibration"], cv_options=calibrator.cv_options)

        calibrator.stereo_rectification = StereoCameraRectificationOptions(**config["stereo_rectification"], cv_options=calibrator.cv_options)
        calibrator.cb_projection = CheckboardProjectionOptions(**config["cb_projection"], cv_options=calibrator.cv_options)
        calibrator.stereo_depth_estimation = DepthEsitmationOptions(
            count=calibrator.stereo_rectification.count, 
            extrinsics_dir=calibrator.stereo_rectification.extrinsic_dir, 
            paired_images_path=calibrator.stereo_rectification.paired_images_path, 
            cv_options=calibrator.cv_options)
        return calibrator

    def calibrate_single_camera(self, opts: SingleCameraCalibrateOptions, capture_images:bool=False, flags: int|None=None, criteria:int|None=None):
        if capture_images:
            if opts.cam_id == self.left_camera_capture.cam_id:
                capture_single_image(self.left_camera_capture)
            else:
                capture_single_image(self.right_camera_capture)
        
        # calibrate left and right cameras
        opts.flags = flags
        opts.criteria = criteria

        single_camera_calibrate(opts)
        
        single_camera_rectification(opts)
    
    def calibrate_stereo_camera(self, capture_images:bool=True, flags:int|None=None, criteria: int|None=None):
        # capture paired images
        if capture_images:
            capture_paired_images(self.stereo_camera_capture)
        
        self.stereo_camera_calibrate.flags = flags
        self.stereo_camera_calibrate.criteria = criteria
        stereo_camera_calibrate(self.stereo_camera_calibrate)
    
    def stereo_rectify(self):
        stereo_rectification(self.stereo_rectification)

    def measure_checkerboard(self) -> Tuple[float, float]:
        return measure_checkerboard(self.cb_projection)
    
    def tune_dispairty(self):
        get_stereo_depth(self.stereo_depth_estimation)
    
    def visualize_checkerboards(self, corners_pcd: np.ndarray|None=None):
        if corners_pcd is None:
            corners_pcd = get_checkerboard_pcd(self.cb_projection)
        # load extrinsinc params for stereo cameras
        extrinsic_params_path = Path(self.stereo_rectification.extrinsic_dir).absolute().joinpath("stereo_calibration.npz")
        extrinsic_params = np.load(extrinsic_params_path)
        R, T = extrinsic_params["R"], extrinsic_params["T"]
        # visualize corners
        viz = CalibrationVisualizer(R=[R], T=[T])
        viz.display_scene(corners_pcd)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
left_single_flags = (cv.CALIB_FIX_TANGENT_DIST)
right_single_flags =  (cv.CALIB_FIX_TANGENT_DIST)
paired_flags = (cv.CALIB_FIX_INTRINSIC + cv.CALIB_ZERO_DISPARITY + cv.CALIB_FIX_TANGENT_DIST)

c = StereoCalibrator.from_yaml("calibrator_config.yaml")
c.calibrate_single_camera(c.left_camera_calibrate, flags=left_single_flags, criteria=criteria)
c.calibrate_single_camera(c.right_camera_calibrate, flags=right_single_flags, criteria=criteria)
c.calibrate_stereo_camera(capture_images=False, flags=paired_flags, criteria=criteria)

c.stereo_rectify()

c.visualize_checkerboards()
corners, dim = c.measure_checkerboard()

c.visualize_checkerboards(corners)

# c.tune_dispairty()
