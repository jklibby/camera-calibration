import pytest

import numpy as np
from pathlib import Path

from calibrator import StereoCalibrator

from .fixtures import calibration_yaml_fixture, config_dict


#inputs
#  calibration_yaml_fixture
#    the yield value at end of calibration_yaml_fixture,
#      which is the temporary yaml file that only exists in this scope
#pseudocode
#  perfroms intrinsic calibration on each camera
#  perfroms extrinsic calibration
#  loads intrinsic and extrinsic parameters from disc 
#  for intrinsic
#    check the shape of intrinsic params, which should be 3x3
#    check if RMSE is below error threshold that we defined in config_dict
#  for extrinsic
#    check the shape of R and T
#    check if RMSE is below error threshold that we defined in config_dict
def test_calibrator(calibration_yaml_fixture):
    calibrator = StereoCalibrator.from_yaml(calibration_yaml_fixture)

    assert_stereo_calibrator_build(calibrator)

    # calibrate left and right camera; do not capture images
    calibrator.calibrate_single_camera(calibrator.left_camera_calibrate)
    calibrator.calibrate_single_camera(calibrator.right_camera_calibrate)

    # assert intrinsic calibration
    assert_intrinsic_calibration(calibrator)

    # calibrate stereo camera
    calibrator.calibrate_stereo_camera()
    assert_extrinsic_calibration(calibrator)

    # get stereo remaps
    calibrator.stereo_rectify()
    assert_stereo_rectification(calibrator)


def assert_stereo_calibrator_build(calibrator: StereoCalibrator):
    assert calibrator != None
    assert calibrator.cv_options != None

    left_single_images_path =str(Path(config_dict["camera_capture"]["image_path"]  + "/{}".format(config_dict["camera_capture"]["left_cam_id"])).absolute())
    right_single_images_path = str(Path(config_dict["camera_capture"]["image_path"]  + "/{}".format(config_dict["camera_capture"]["right_cam_id"])).absolute())
    single_camera_images_path = str(Path(config_dict["camera_capture"]["image_path"]).absolute())
    stereo_images_path = str(Path(config_dict["camera_capture"]["stereo_images_path"]).absolute())
    intrinsics_path = str(Path(config_dict["camera_calibration"]["intrinsic_params_dir"]).absolute())
    extrinsics_path = str(Path(config_dict["camera_calibration"]["extrinsic_params_dir"]).absolute())

    assert calibrator.left_camera_capture != None
    assert calibrator.left_camera_capture.cam_id == config_dict["camera_capture"]["left_cam_id"]
    assert calibrator.left_camera_capture.count == config_dict["camera_capture"]["image_count"]
    assert calibrator.left_camera_capture.path == left_single_images_path

    assert calibrator.right_camera_capture != None
    assert calibrator.right_camera_capture.cam_id == config_dict["camera_capture"]["right_cam_id"]
    assert calibrator.right_camera_capture.count == config_dict["camera_capture"]["image_count"]
    assert calibrator.right_camera_capture.path == right_single_images_path

    assert calibrator.stereo_camera_capture != None
    assert calibrator.stereo_camera_capture.count == config_dict["camera_capture"]["stereo_images_count"]
    assert calibrator.stereo_camera_capture.path == stereo_images_path
    assert calibrator.stereo_camera_capture.left_cam_id == config_dict["camera_capture"]["left_cam_id"]
    assert calibrator.stereo_camera_capture.right_cam_id == config_dict["camera_capture"]["right_cam_id"]

    assert calibrator.left_camera_calibrate != None
    assert calibrator.left_camera_calibrate.cam_id == config_dict["camera_capture"]["left_cam_id"]
    assert calibrator.left_camera_calibrate.count == config_dict["camera_capture"]["image_count"]
    assert calibrator.left_camera_calibrate.path == single_camera_images_path
    assert calibrator.left_camera_calibrate.dir == intrinsics_path
    assert calibrator.left_camera_calibrate.headless == config_dict["camera_calibration"]["headless"]
    assert calibrator.left_camera_calibrate.error_threshold == config_dict["camera_calibration"]["single_calibration_error_threshold"]
    assert calibrator.left_camera_calibrate.pattern_size == config_dict["camera_calibration"]["pattern_size"]

    assert calibrator.right_camera_calibrate != None
    assert calibrator.right_camera_calibrate.cam_id == config_dict["camera_capture"]["right_cam_id"]
    assert calibrator.right_camera_calibrate.count == config_dict["camera_capture"]["image_count"]
    assert calibrator.right_camera_calibrate.path == single_camera_images_path
    assert calibrator.right_camera_calibrate.dir == intrinsics_path
    assert calibrator.right_camera_calibrate.headless == config_dict["camera_calibration"]["headless"]
    assert calibrator.right_camera_calibrate.error_threshold == config_dict["camera_calibration"]["single_calibration_error_threshold"]
    assert calibrator.right_camera_calibrate.pattern_size == config_dict["camera_calibration"]["pattern_size"]

    assert calibrator.stereo_camera_calibrate != None
    assert calibrator.stereo_camera_calibrate.count == config_dict["camera_capture"]["image_count"]
    assert calibrator.stereo_camera_calibrate.intrinsics_dir == intrinsics_path
    assert calibrator.stereo_camera_calibrate.dir == extrinsics_path
    assert calibrator.stereo_camera_calibrate.headless == config_dict["camera_calibration"]["headless"]
    assert calibrator.stereo_camera_calibrate.error_threshold == config_dict["camera_calibration"]["stereo_calibration_error_threshold"]
    assert calibrator.stereo_camera_calibrate.pattern_size == config_dict["camera_calibration"]["pattern_size"]

    assert calibrator.stereo_rectification != None
    assert calibrator.stereo_rectification.paired_images_path == stereo_images_path
    assert calibrator.stereo_rectification.count == config_dict["camera_capture"]["image_count"]
    assert calibrator.stereo_rectification.headless == config_dict["camera_calibration"]["headless"]
    assert calibrator.stereo_rectification.left_cam_id == config_dict["camera_capture"]["left_cam_id"]
    assert calibrator.stereo_rectification.right_cam_id == config_dict["camera_capture"]["right_cam_id"]

    assert calibrator.cb_projection != None
    assert calibrator.cb_projection.pattern_size == config_dict["camera_calibration"]["pattern_size"]
    assert calibrator.cb_projection.validation_pattern_size == config_dict["validation"]["pattern_size"]
    assert calibrator.cb_projection.world_scaling == config_dict["camera_calibration"]["world_scaling"]

    assert calibrator.stereo_depth_estimation != None
    assert calibrator.stereo_depth_estimation.extrinsics_dir == extrinsics_path
    assert calibrator.stereo_depth_estimation.paired_images_path == stereo_images_path
    
def assert_intrinsic_calibration(calibrator: StereoCalibrator):
    left_cam_id, right_cam_id = calibrator.left_camera_calibrate.cam_id, calibrator.right_camera_calibrate.cam_id
    intrinsics_path = Path(config_dict["camera_calibration"]["intrinsic_params_dir"]).absolute()
    left_intrinsics_path = str(intrinsics_path.joinpath("camera_calibration_{}.npz".format(left_cam_id)).absolute())
    right_intrinsics_path = str(intrinsics_path.joinpath("camera_calibration_{}.npz".format(right_cam_id)).absolute())
    
    intrinsic_data_0 = np.load(left_intrinsics_path)
    intrinsic_data_1 = np.load(right_intrinsics_path)

    lrmse = intrinsic_data_0["RMSE"]
    rrmse = intrinsic_data_1["RMSE"]

    k_l = intrinsic_data_0["calibration_mtx"]
    k_r = intrinsic_data_1["calibration_mtx"]

    assert lrmse <= calibrator.left_camera_calibrate.error_threshold
    assert rrmse <= calibrator.right_camera_calibrate.error_threshold

    assert k_l.shape == (3, 3)
    assert k_r.shape == (3, 3)

def assert_extrinsic_calibration(calibrator: StereoCalibrator):
    extrinsics_path = Path(config_dict["camera_calibration"]["extrinsic_params_dir"]).joinpath("stereo_calibration.npz").absolute()

    extrinsics = np.load(extrinsics_path)
    R, T, rmse = extrinsics["R"], extrinsics["T"], extrinsics["RMSE"]

    assert R.shape == (3, 3)
    assert T.shape == (3, 1)
    assert rmse <= calibrator.stereo_camera_calibrate.error_threshold

def assert_stereo_rectification(calibrator: StereoCalibrator):
    left_map_x, left_map_y, right_map_x, right_map_y = calibrator.cb_projection.load_remaps()

    assert left_map_x.all() == True
    assert left_map_y.all() == True
    assert right_map_x.all() == True
    assert right_map_y.all() == True
