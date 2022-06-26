import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def interactive_visulization(points, molecules, labels, qcharges, rmsd):
    """
    points: [num_frames, 2]
    molecules: [num_frames, 3, num_atoms]
    """

    # Preparing figure object 
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "xy"}, {"type": "scene"}]],
    )
    
    points_plot = go.Scatter(
        x=points[:, 0],
        y=points[:, 1],
        hovertext=list(map(str, zip(np.arange(points.shape[0]), qcharges, rmsd))),
        #hovertext=np.arange( )
        mode="markers",
        name="latent_points",
        marker=dict(
            color=labels,
        )
    )

    molecule_3d_plot = go.Scatter3d(
        x = molecules[0, 0, :],
        y = molecules[0, 1, :],
        z = molecules[0, 2, :],
        mode="markers",
        marker=dict(
            size=3,
        ),
        name="molecule",
    )

    fig.add_trace(points_plot, row=1, col=1)
    fig.add_trace(molecule_3d_plot, row=1, col=2)

    layout = fig.layout
    # layout["xaxis"]["range"] = [-2.2, 2.2]
    # layout["yaxis"]["range"] = [-2.2, 2.2]
    # layout["zaxis"]["range"] = [-2.2, 2.2]

    # Preparing interactive widget
    widget = go.FigureWidget([points_plot, molecule_3d_plot], layout=layout)
    # widget.layout.hovermode = 'closest'

    points_plot = widget.data[0]
    molecule_3d_plot = widget.data[1]

    # Defining callback with 3 parameters: trace = graphical object with some changes, points and selector
    def callback(trace, points, selector):
        # A list with the size of the points, all of them have the same size
        # s = [5] * N # list(points_plot.marker.size)

        # Iteration over indexes, we want only one of the points
        for i in points.point_inds:
            # Imitating selection of the particular point
            # s[i] = 10

            # Go into batch_update for buffering
            with widget.batch_update():
                # coordinates of i frame
                molecule_3d_plot.x = molecules[i, 0, :]
                molecule_3d_plot.y = molecules[i, 1, :]
                molecule_3d_plot.z = molecules[i, 2, :]

                # changing sizes of all the points, it is necessary to make another list to make it work 
                # points_plot.marker.size = s

            return 

    # registring callback, possible modes (on_cxlick), (on_selection), etc
    points_plot.on_hover(callback)

    # Widget visualization
    return widget