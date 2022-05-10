import cv2
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Visualizer():
    """Helper class to visualize depth and segmentation prediction.
    """
    def __init__(self, width, height, img, pred_depth, pred_seg):
        self.fig = go.FigureWidget(
            make_subplots(
                rows=2, cols=2, 
                column_widths=[0.4, 0.6], 
                specs = [
                    [{'type': 'image'}, {'type': 'surface'}],
                    [{'type': 'xy'}, {'type': 'surface'}]
                ]
            )
        )
        self.camera = {
            'center': {'x': 0, 'y': 0, 'z': 0},
            'eye': {'x': -0.05060184209509067, 'y': -0.8272060402743945, 'z': -2.0001673981220023},
            'projection': {'type': 'perspective'},
            'up': {'x': 0, 'y': 0, 'z': 1}
        }
        self._visualize_init(img, pred_depth, pred_seg)
        self.fig.update_layout(
            width = width,
            height = height
        )
        self.data = self.fig.data

    def update_img(self, img):
        self.data[0].z = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def update_depth(self, depth, img):
        self.data[1].z = depth
        self.data[3].z = depth
        self.data[3].update(surfacecolor=img[:,:,0])

    def update_seg(self, seg):
        self.data[2].z = np.rot90(seg.squeeze().T, 1)

    def _visualize_init(self, img, pred_depth, pred_seg):
        self.fig.add_trace(go.Image(z=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), row = 1, col = 1)
        self.fig.add_trace(
            go.Surface(
                z = pred_depth.squeeze(),
                showscale = False
            ),
            row=1, col=2
        )
        self.fig.add_trace(
            go.Contour(
                z = np.rot90(pred_seg.squeeze().T, 1),
                colorscale='emrld',
                showscale = False
            ),
            row=2, col=1
        )
        self.fig.add_trace(
            go.Surface(
                z = pred_depth.squeeze(),
                colorscale='viridis', 
                showscale = False
            ),
            row=2, col=2
        )

        self.fig.layout.scene.camera = self.camera
        self.fig.layout.scene2.camera = self.camera
        self.fig.data[3].update(surfacecolor=img[:,:,0])
