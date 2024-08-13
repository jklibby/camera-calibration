import abc
from typing import *
import os
import yaml
from pathlib import Path
import numpy as np
import cv2 as cv

from core import capture_single_image, capture_paired_images
from core import single_camera_calibrate, stereo_camera_calibrate, stereo_rectification, single_camera_rectification
from options import CVOptionType, CameraCaptureType, CameraCalibrationType
from options import CVCameraOptions, StereoCameraCalibrationOptions, StereoCameraRectificationOptions
from options import SingleCameraCaptureOptions, SingleCameraCalibrateOptions, StereoCameraCaptureOptions
from options import CheckboardProjectionOptions, Validation, DepthEstimationOptions
from projection import measure_checkerboard, get_checkerboard_pcd
from visualization import MatplotlibCalibrationVisualizer as CalibrationVisualizer
from depth_estimation.stereo_depth import get_stereo_depth, get_live_stereo_depth


class CalibratorConfig(TypedDict):
    cv_options: CVOptionType
    camera_capture: CameraCaptureType
    camera_calibration: CameraCalibrationType
    validation: Validation

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
    stereo_depth_estimation: DepthEstimationOptions

    cb_projection: CheckboardProjectionOptions

    @staticmethod
    def from_yaml(file_path: str) -> 'StereoCalibrator':
        config: CalibratorConfig = load_yaml(file_path)
        return StereoCalibrator.from_config(config)
    
    @staticmethod
    def from_config(config: CalibratorConfig) -> 'StereoCalibrator':
        calibrator = StereoCalibrator()
        calibrator.cv_options = CVCameraOptions(**config["cv_options"])
        calibrator.left_camera_capture = SingleCameraCaptureOptions(
            cv_options=calibrator.cv_options, 
            cam_id=config["camera_capture"]["left_cam_id"], 
            image_path=config["camera_capture"]["image_path"], 
            image_count=config["camera_capture"]["image_count"]
        )
        calibrator.right_camera_capture = SingleCameraCaptureOptions(
            cv_options=calibrator.cv_options, 
            cam_id=config["camera_capture"]["right_cam_id"], 
            image_path=config["camera_capture"]["image_path"], 
            image_count=config["camera_capture"]["image_count"]
        )
        calibrator.left_camera_calibrate = SingleCameraCalibrateOptions(
            cv_options=calibrator.cv_options, 
            cam_id=config["camera_capture"]["left_cam_id"],
            count=config["camera_capture"]["image_count"],
            image_path=config["camera_capture"]["image_path"], 
            pattern_size=config["camera_calibration"]["pattern_size"],
            intrinsic_params_dir=config["camera_calibration"]["intrinsic_params_dir"],
            headless=config["camera_calibration"]["headless"],
            error_threshold=config["camera_calibration"]["single_calibration_error_threshold"]
        )
        calibrator.right_camera_calibrate = SingleCameraCalibrateOptions(
            cv_options=calibrator.cv_options, 
            cam_id=config["camera_capture"]["right_cam_id"],
            count=config["camera_capture"]["image_count"],
            image_path=config["camera_capture"]["image_path"], 
            pattern_size=config["camera_calibration"]["pattern_size"],
            intrinsic_params_dir=config["camera_calibration"]["intrinsic_params_dir"],
            headless=config["camera_calibration"]["headless"],
            error_threshold=config["camera_calibration"]["single_calibration_error_threshold"]
        )
        
        calibrator.stereo_camera_capture = StereoCameraCaptureOptions(
            cv_options=calibrator.cv_options, 
            left_cam_id=config["camera_capture"]["left_cam_id"], 
            right_cam_id=config["camera_capture"]["right_cam_id"], 
            count=config["camera_capture"]["stereo_images_count"],
            path=config["camera_capture"]["stereo_images_path"],
        )
        calibrator.stereo_camera_calibrate = StereoCameraCalibrationOptions(
            cv_options=calibrator.cv_options, 
            left_cam_id=config["camera_capture"]["left_cam_id"], 
            right_cam_id=config["camera_capture"]["right_cam_id"], 
            count=config["camera_capture"]["stereo_images_count"], 
            image_path=config["camera_capture"]["stereo_images_path"],
            intrinsic_params_dir=config["camera_calibration"]["intrinsic_params_dir"],
            pattern_size=config["camera_calibration"]["pattern_size"],
            extrinsic_params_dir=config["camera_calibration"]["extrinsic_params_dir"],
            headless=config["camera_calibration"]["headless"],
            error_threshold=config["camera_calibration"]["stereo_calibration_error_threshold"],
        )

        calibrator.stereo_rectification = StereoCameraRectificationOptions(
            cv_options=calibrator.cv_options, 
            left_cam_id=config["camera_capture"]["left_cam_id"], 
            right_cam_id=config["camera_capture"]["right_cam_id"], 
            count=config["camera_capture"]["stereo_images_count"], 
            image_path=config["camera_capture"]["stereo_images_path"],
            intrinsic_params_dir=config["camera_calibration"]["intrinsic_params_dir"],
            extrinsic_params_dir=config["camera_calibration"]["extrinsic_params_dir"],
            headless=config["camera_calibration"]["headless"],
        )
        calibrator.cb_projection = CheckboardProjectionOptions(
            cv_options=calibrator.cv_options, 
            left_cam_id=config["camera_capture"]["left_cam_id"], 
            right_cam_id=config["camera_capture"]["right_cam_id"], 
            count=config["camera_capture"]["stereo_images_count"], 
            paired_images_path=config["camera_capture"]["stereo_images_path"],
            intrinsic_params_dir=config["camera_calibration"]["intrinsic_params_dir"],
            pattern_size=config["camera_calibration"]["pattern_size"],
            extrinsic_params_dir=config["camera_calibration"]["extrinsic_params_dir"],
            headless=config["camera_calibration"]["headless"],
            world_scaling=config["camera_calibration"]["world_scaling"], 
            validation_pattern_size=config["validation"]["pattern_size"],
        )
        calibrator.stereo_depth_estimation = DepthEstimationOptions(
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
        
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        flags =  (cv.CALIB_FIX_TANGENT_DIST)
        
        # calibrate left and right cameras
        opts.flags = flags
        opts.criteria = criteria

        _, corners_3d = single_camera_calibrate(opts)
        corners_3d = np.array(corners_3d)
        
        single_camera_rectification(opts)

        if not opts.headless:
            viz = CalibrationVisualizer(cams=1)
            viz.display_scene(corners_3d)
    
    def calibrate_stereo_camera(self, capture_images:bool=False, flags:int|None=None, criteria: int|None=None):
        # capture paired images
        if capture_images:
            capture_paired_images(self.stereo_camera_capture)
        
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        flags = (cv.CALIB_FIX_INTRINSIC + cv.CALIB_ZERO_DISPARITY + cv.CALIB_FIX_TANGENT_DIST)
        
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

