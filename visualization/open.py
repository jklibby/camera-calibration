import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class Open3DCalibrationVisualizer():
    def __init__(self, cams=2, R=[], T=[]):
        self.cams = cams
        self.R = R
        self.T = T
        
        self.camera_positions = [np.array([0, 0, 0], dtype=np.float32)]
        for r, t in zip(self.R, self.T):
            camera_pos = -t.flatten()
            self.camera_positions.append(camera_pos)
        self.viz: o3d.visualization.Visualizer = o3d.visualization.Visualizer()


    def display_scene(self, corners, **kwargs):
        "creates an Open3DE scene with cameras"
        # convert camera coordinates to point clouds
        self.viz.create_window()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.camera_positions)
        self.viz.add_geometry(pcd)
        
        for cam_pos in self.camera_positions:
            cam_line_pcd = camera_lineset(cam_pos, 3, 3)
            self.viz.add_geometry(cam_line_pcd)

        cb_lineset, cb_pcd = checkerboard_lineset(corners)
        self.viz.add_geometry(cb_lineset)
        self.viz.add_geometry(cb_pcd)

        self.viz.run()



def camera_lineset(center, w, h):
    camera_plane = np.array([center] * 4)
    scaling = np.array([
        [-1*h/2, -1*w/2, 0],
        [-1*h/2, +1*w/2, 0],
        [+1*h/2, +1*w/2, 0],
        [+1*h/2, -1*w/2, 0],
    ])
    camera_plane += scaling

    tunnel_plane = np.array([center] * 4) + scaling * 0.5
    tunnel_plane[:, 2] = -1.5 + center[2]

    tunnel_plane_2 = tunnel_plane.copy()
    tunnel_plane_2[:, 2] = -2 + center[2]

    camera_points = np.vstack([camera_plane, tunnel_plane, tunnel_plane_2])
    camera_lines = list()
    for i in range(3):
        for j in range(4):
            if j == 3:
                camera_lines.append((j + 4*i, i*4))
            else:
                camera_lines.append((j + 4*i, j + 4*i + 1))

            if i > 0:
                camera_lines.append((j+(i-1)*4, j+ i*4))
    color = np.random.rand(3)
    colors = o3d.utility.Vector3dVector([color] * len(camera_lines))
    camera_lines = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(camera_points), lines=o3d.utility.Vector2iVector(camera_lines))
    camera_lines.colors = colors
    return camera_lines


def checkerboard_lineset(corners):
    # created points
    print(corners.shape)
    num_corners = corners.shape[1]
    corners = corners.reshape(-1, 3)
    colors = color_bar(len(corners))
    corners = o3d.utility.Vector3dVector(corners)

    corner_pcd = o3d.geometry.PointCloud(corners)
    corner_pcd.colors = colors
    
    # create lines
    lines = list()
    for i in range(1, len(corners)):
        if i % num_corners == 0:
            continue
        lines.append((i-1, i))
    lines = o3d.utility.Vector2iVector(lines)
    # create color map
    

    # return checkboard lineset
    cb_lineset = o3d.geometry.LineSet(points=corners, lines=lines)
    cb_lineset.colors = color_bar(len(lines))
    return [cb_lineset, corner_pcd]

def color_bar(length):
    colormap = cm.get_cmap('viridis')
    colormap(0, 1)
    gradient = np.linspace(0, 1, length)
    color_gradient = np.array([list(colormap(i))[:3] for i in gradient])
    return o3d.utility.Vector3dVector(color_gradient)

