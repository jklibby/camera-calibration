import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class MatplotlibCalibrationVisualizer():
    def __init__(self, cams=2, R=[], T=[]):
        self.cams = cams
        self.R = R
        self.T = T
        
        self.camera_positions = [np.array([0, 0, 0], dtype=np.float32)]
        for r, t in zip(self.R, self.T):
            
            camera_pos = t.flatten()
            self.camera_positions.append(camera_pos)
        
        self.camera_positions = np.array(self.camera_positions)


    def display_scene(self, corners, **kwargs):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # plot cameras
        camera_colors = color_bar(len(self.camera_positions))
        for index, cp in enumerate(self.camera_positions):
            cc, camera_l = camera_lineset(cp, 2, 2)
            for lc in camera_l:
                ax.plot(lc[:, 0], lc[:, 1], lc[:, 2], c=camera_colors[index])
        ax.scatter(self.camera_positions[:, 0], self.camera_positions[:, 1], self.camera_positions[:, 2])

        # plot checkerboard
        cbb, lines, colors, l_colors = checkerboard_lineset(corners)
        for index, ll in enumerate(lines):
            ax.plot(ll[:, 0], ll[:, 1], ll[:, 2], c=l_colors[index])
        ax.scatter(cbb[:, 0], cbb[:, 1], cbb[:, 2], c=colors)

        plt.show()

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
                camera_lines.append((camera_points[j + 4*i], camera_points[i*4]))
            else:
                camera_lines.append((camera_points[j + 4*i], camera_points[j + 4*i + 1]))

            if i > 0:
                camera_lines.append((camera_points[j+(i-1)*4], camera_points[j+ i*4]))
    return camera_points, np.array(camera_lines)
    

def checkerboard_lineset(corners):
    # created points
    num_corners = corners.shape[1]
    corners = corners.reshape(-1, 3)
    colors = color_bar(len(corners))

    # create lines
    lines = list()
    l_colors = list()
    for i in range(1, len(corners)):
        if i % num_corners == 0:
            continue
        lines.append((corners[i-1], corners[i]))
        l_colors.append(colors[i])
    lines = np.array(lines)
    return corners, lines, colors, l_colors

    
def color_bar(length):
    colormap = cm.get_cmap('viridis')
    colormap(0, 1)
    gradient = np.linspace(0, 1, length)
    color_gradient = [colormap(i) for i in gradient]
    return color_gradient
