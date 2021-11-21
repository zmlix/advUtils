from os import name
import plotly.graph_objects as go


class PlotUtils():
    def __init__(self) -> None:
        pass

    def plot3d(self, img, width=1000, height=800, scale=1):
        img_b = img.transpose(2, 0, 1) * scale
        # print(utils.summary(img_b[0][:][:]),img_b[0][:][:].shape,img_b)
        # print(img_b[0][30][30] * 255, img_b[1][30][30] * 255, img_b[2][30][30] * 255)
        fig = go.Figure(data=[
            go.Surface(z=img_b[0][::-1][:], opacity=0.5, name='Red'),
            go.Surface(z=img_b[1][::-1][:] + 10,
                       showscale=False,
                       opacity=0.5,
                       name="Green"),
            go.Surface(z=img_b[2][::-1][:] + 20,
                       showscale=False,
                       opacity=0.5,
                       name="Blue"),
        ])
        fig.update_traces(contours_z=dict(show=True,
                                          usecolormap=True,
                                          highlightcolor="limegreen",
                                          project_z=True))
        fig.update_layout(scene=dict(
            xaxis=dict(
                nticks=4,
                range=[0, 224],
            ),
            yaxis=dict(
                nticks=4,
                range=[0, 224],
            ),
            zaxis=dict(
                nticks=4,
                range=[-10, 30],
            ),
        ),
                          width=width,
                          height=height,
                          margin=dict(r=20, l=20, b=20, t=20))
        fig.show()