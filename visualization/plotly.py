import numpy as np
import plotly.graph_objects as go
import plotly.express as px

class PlotlyCalibrationVisualizer():
    def __init__(self, cams=2, R=[], T=[]):
        self.cams = cams
        self.R = R
        self.T = T
        
        self.camera_positions = [np.array([0, 0, 0], dtype=np.float32)]
        for r, t in zip(self.R, self.T):
            camera_pos = -t.flatten()
            self.camera_positions.append(camera_pos)
        
        self.camera_positions = np.array(self.camera_positions)

    def display_scene(self, corners, display_magnitudes=False):
        # Create a 3D figure using Plotly's graph_objects
        fig = go.Figure()

        # plot cameras
        camera_colors = color_bar(len(self.camera_positions))
        for index, cp in enumerate(self.camera_positions):
            cc, camera_l = camera_lineset(cp, 2, 2)
            for lc in camera_l:
                fig.add_trace(go.Scatter3d(
                    x=lc[:, 0], y=lc[:, 1], z=lc[:, 2],
                    mode='lines',
                    line=dict(color=camera_colors[index]),
                    name=f'Camera {index}'
                ))
        fig.add_trace(go.Scatter3d(
            x=self.camera_positions[:, 0],
            y=self.camera_positions[:, 1],
            z=self.camera_positions[:, 2],
            mode='markers',
            marker=dict(size=5, color='red'),
            name='Camera Positions'
        ))

        # plot checkerboard
        cbb, lines, colors, l_colors, magnitude = checkerboard_lineset(corners)
        for index, ll in enumerate(lines):
            fig.add_trace(go.Scatter3d(
                x=ll[:, 0], y=ll[:, 1], z=ll[:, 2],
                mode='lines+text',
                line=dict(color=l_colors[index]),
                text=magnitude[index] if display_magnitudes else "",
                name=f'Object Line {index}'
            ))
        fig.add_trace(go.Scatter3d(
            x=cbb[:, 0],
            y=cbb[:, 1],
            z=cbb[:, 2],
            mode='markers',
            marker=dict(size=5, color='blue'),
            name='Checkerboard Corners'
        ))

        # Update layout for better visualization
        fig.update_layout(
            scene=dict(
                xaxis_title="X Axis",
                yaxis_title="Z Axis",
                zaxis_title="Y Axis",
                aspectmode="data"
            ),
            title="3D Camera and Checkerboard Visualization"
        )
        
        fig.show()

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
    num_corners = corners.shape[1]
    corners = corners.reshape(-1, 3)
    colors = color_bar(len(corners))

    # Create lines between consecutive corners
    lines = []
    l_colors = []
    magnitude = list()
    for i in range(1, len(corners)):
        if i % num_corners == 0:
            continue
        lines.append((corners[i-1], corners[i]))
        l_colors.append(colors[i])
    
    for index, line in enumerate(lines):
        mag = np.linalg.norm(line[1] - line[0])    
        print(line, mag)
        magnitude.append(["start", "{:0.3f} cm".format(mag)])
    lines = np.array(lines)
    return corners, lines, colors, l_colors, magnitude

def color_bar(length):
    colormap = px.colors.sequential.Viridis
    gradient = np.linspace(0, 1, length)
    color_gradient = [colormap[int(i * (len(colormap) - 1))] for i in gradient]
    return color_gradient
