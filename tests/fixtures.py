import pytest
from pathlib import Path
import yaml

from calibrator import CalibratorConfig

config_dict: CalibratorConfig = {
    'cv_options': {
        'resolution': [1920, 1080],
        'window_size': [1280, 720]
    },
    'camera_capture': {
        'left_cam_id': 0,
        'right_cam_id': 1, 
        'image_count': 50, 
        'image_path': 'test_images/calibration-images/calibration-images/single_images/',
        'stereo_images_path': 'test_images/calibration-images/calibration-images/stereo_images/', 
        'stereo_images_count': 50
    },
    'camera_calibration': {
        'intrinsic_params_dir': 'test_data/intrinsic_dir',
        'extrinsic_params_dir': 'test_data/extrinsic_dir',
        'pattern_size': [10, 7],
        'world_scaling': 1.5,
        'single_calibration_error_threshold': 0.3,
        'stereo_calibration_error_threshold': 0.7, 
        'headless': True
    },
    'validation': {
        'pattern_size': [10, 7]
    }
}

##when pytest gets called, it will fist call fixturs.py
##  and then calibration_yaml_fixture will be called
##  then, later, when test_* gets called and calibration_yaml_fixture
#     is passed in as it's parameter, it will return the yaml_file
#     which is yielded here at the end of this function.
#   The whole point of this is so that we can have this yaml file
#     which only exists temporarily in this session.
#   The values of that yaml file ar as defined in config_dict
@pytest.fixture(scope="session")
def calibration_yaml_fixture(tmp_path_factory):
    # dump config yaml into a tmp file
    yaml_data = yaml.dump(config_dict)
    yaml_file:Path = tmp_path_factory.mktemp("config") / "test_calibrator_config.yaml"
    with open(yaml_file, "+w") as f:
        f.write(yaml_data)
        f.close()
    yield yaml_file
    
