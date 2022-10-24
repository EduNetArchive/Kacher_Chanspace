import plotly as py
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def three_img(points, z, color):
    x = points[:, 0]
    y = points[:, 1]
    z = z

    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=1,
            color=color,                # set color to an array/list of desired values
            colorscale='Viridis',   # choose a colorscale
            opacity=0.5
        )
    )])

    # tight layout
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()