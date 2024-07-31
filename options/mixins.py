import numpy as np
from pathlib import Path

class SingleImageLoader:
    def load_single_images(self):
        path = Path(self.path)
        image_paths = [str(path.joinpath("{}".format(self.cam_id), 'camera{}_{}.png'.format(self.cam_id, cap_count)).absolute()) for cap_count in range(self.count)]
        return image_paths


class PairedImageLoader:
    def load_paired_images(self):
        path = Path(self.paired_images_path)
        image_paths = [(str(path.joinpath('camera{}_{}.png'.format(self.left_cam_id, cap_count)).absolute()), str(path.joinpath('camera{}_{}.png'.format(self.right_cam_id, cap_count)).absolute())) for cap_count in range(self.count)]
        return image_paths

class StereoRectificationMapLoader:
    def load_remaps(self):
        path = str(Path(self.extrinsics_dir).joinpath("stereo_rectification", "stereo_rectification_maps.npz").absolute())
        remaps = np.load(path)
        left_map_x, left_map_y, right_map_x, right_map_y = remaps['left_map_x'], remaps['left_map_y'], remaps['right_map_x'], remaps['right_map_y']
        return left_map_x, left_map_y, right_map_x, right_map_y
