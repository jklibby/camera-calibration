"""
options

A package for options to be passed in various steps of the stereo camera calibration pipeline.

Modules:
    - core: options needed for core calibration operations, camera capture, calibration and rectification.
    - depth_estimation: Options needed for creating a depth map.
    - projection: options needed for validation of calibration. 
    - option_types: Typed Dicts for loading data from config file. 
"""

from .core import CVCameraOptions, SingleCameraCaptureOptions, SingleCameraCalibrateOptions, StereoCameraCaptureOptions, StereoCameraCalibrationOptions, StereoCameraRectificationOptions
from .projection import CheckboardProjectionOptions

from .option_types import CVOptionType
from .option_types import CameraCaptureType, CameraCalibrationType
from .option_types import Validation

from .depth_estimation import DepthEstimationOptions
