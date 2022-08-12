import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def interactive_visulization(points, labels, qcharges=None, rmsd=None, molecules=None, bg=None, bb=None):
    """
    points: [num_frames, 2]
    molecules: [num_frames, 3, num_atoms]
    """

    # Preparing figure object 
    if molecules is not None:
        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "xy"}, {"type": "scene"}]],
        )
    else: 
        fig = make_subplots(
            rows=1,
            cols=1,
            specs=[[{"type": "xy"}]],
        )
    if rmsd is not None and qcharges is not None: 
        hovertext = list(map(str, zip(np.arange(points.shape[0]), qcharges, rmsd)))
    elif qcharges is not None:
        hovertext = list(map(str, zip(np.arange(points.shape[0]), qcharges)))
    elif rmsd is not None:
        hovertext = list(map(str, zip(np.arange(points.shape[0]), rmsd)))
    else: 
        hovertext = np.arange(points.shape[0])


        
    points_plot = go.Scatter(
        x=points[:, 0],
        y=points[:, 1],
        hovertext=hovertext,
        #hovertext=np.arange( )
            mode="markers",
            name="latent_points",
            marker=dict(
                color=labels,
            )

    )
    

    fig.add_trace(points_plot, row=1, col=1)
    
    if molecules is not None:
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


        fig.add_trace(molecule_3d_plot, row=1, col=2)

    layout = fig.layout
        # layout["xaxis"]["range"] = [-2.2, 2.2]
        # layout["yaxis"]["range"] = [-2.2, 2.2]
        # layout["zaxis"]["range"] = [-2.2, 2.2]
    
    if bg is not None:
        from PIL import Image
        from matplotlib import cm

        energy_array = np.load(bg)
        # energy_img = Image.fromarray(energy_array)
        energy_array = (energy_array - energy_array.min()) / \
                       (energy_array.max() - energy_array.min())

        energy_img = Image.fromarray(np.uint8(cm.viridis(energy_array)*255))
        # from matplotlib import pyplot as plt
        # plt.imshow(np.array(energy_img))
        # plt.show()
        
        x1,y1,x2,y2 = np.load(bb)
        fig.add_layout_image(
        dict(
            source=energy_img,
            #xref="x",
            #yref="y",
            x=x1,
            y=y2,
            sizex=x2-x1,
            sizey=y2-y1,
            sizing="stretch",
            opacity=0.5
            )
        #layer="below")
        )   
    # fig.update_layout(template="plotly_white")
    fig.update_layout(
        width = 400,
        height = 1000,
    )
    # Preparing interactive widget
    
    if molecules is not None:
        widget = go.FigureWidget([points_plot, molecule_3d_plot], layout=layout)
        molecule_3d_plot = widget.data[1]
    else:
        widget = go.FigureWidget([points_plot], layout=layout)
    
    points_plot = widget.data[0]
       
    # widget.layout.hovermode = 'closest'

    

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
    if molecules is not None:
        points_plot.on_hover(callback)

    # Widget visualization
    return widget