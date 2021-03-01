import os
import math
import numpy as np
from geopandas import GeoDataFrame
from shapely.geometry import Point
import matplotlib.pyplot as plt
import framework.tools as tools
from framework.windfield import Windfield
from framework.windfield import WindDataFrame
import pandas as pd


def setup_plot(map: GeoDataFrame, title=None, resolution=100):
    min_x, min_y, max_x, max_y = map['geometry'].total_bounds

    if (max_x - min_x > max_y - min_y):
        padding = (max_x - min_x) / 10
        nof_x_values = resolution
        nof_y_values = round(nof_x_values * (max_y - min_y) / (max_x - min_x))#.astype(np.int64)
    else:
        padding = (max_y - min_y) / 10
        nof_y_values = resolution
        nof_x_values = round(nof_y_values * (max_x - min_x) / (max_y - min_y))#.astype(np.int64)

    x0 = min_x - padding
    x1 = max_x + padding
    y0 = min_y - padding
    y1 = max_y + padding

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, aspect='equal', label='Foo', autoscale_on=False, xlim=(x0, x1), ylim=(y0, y1))
    ax.title.set_text(title)

    return fig, ax, x0, x1, nof_x_values, y0, y1, nof_y_values


def plot_wind_vector(ax, x, y, u, v, scale, arrow_head_width, color='red'):
    dx = u * scale
    dy = v * scale
    ax.arrow(x, y, dx, dy, facecolor=color, edgecolor=color, head_width=arrow_head_width)
    return True


def plot_wind_field(map: GeoDataFrame,
                    calibration_data: WindDataFrame,
                    windfield: Windfield,
                    title=None,
                    plot_calibration_data=True,
                    mask=None):
    """ Plots the given wind field on a map.

    Plots the given wind field as a vector field flow on the given map.
    Optionally, the calibration data can also plotted as arrows.

    Parameters
    ----------
    map: GeoDataFrame
        The map onto which the wind field is plotted
    calibration_data: WindDataFrame
        The original observation data
    windfield: Windfield
        The windfield to plot
    title: string, optional
        The title of the plot, default is None
    plot_calibration_data: boolean, optional
        Whether to plot the calibration data as arrows, default is True
    """

    # Setup
    fig, ax, x0, x1, xn, y0, y1, yn = setup_plot(map=map, title=title)
    scale = max(x1 - x0, y1 - y0) / 150
    arrow_head_width = max(x1 - x0, y1 - y0) / 100

    # Plot background map as black contour
    map.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)

    # Plot calibration data as red arrows
    if (plot_calibration_data):
        calibration_data.apply(lambda row: plot_wind_vector(ax=ax, x=row.x, y=row.y, u=row.u, v=row.v, scale=scale, arrow_head_width=arrow_head_width), axis=1)

    # Plot windfield as a vector field flow
    Y, X = np.mgrid[y1:y0:yn*1j, x0:x1:xn*1j]

    x = X.flatten()
    y = Y.flatten()

    wind = windfield.predict(pd.Series(x), pd.Series(y))

    u = wind['u']
    v = wind['v']

    U = np.array(u).reshape(np.shape(X))
    V = np.array(v).reshape(np.shape(X))

    speed = np.sqrt(U * U + V * V)
    linewidth = 5 * speed / speed.max() if (speed.max() != 0) else np.zeros_like(speed)

    if mask is not None:
        m, n = X.shape
        zero_speed = np.zeros((m,n))
        points = [Point(x,y) for (x,y) in zip(x,y)]
        f_points = [mask.contains(p).any() for p in points]
        f_points = np.reshape(f_points, (m,n))
        U = np.where(f_points, U, zero_speed)
        V = np.where(f_points, V, zero_speed)

    ax.streamplot(X, Y, U, V, color='blue', linewidth=linewidth)
    return fig


def plot_wind_field_at_points(
    map: GeoDataFrame,
    calibration_data: WindDataFrame,
    prediction_data: WindDataFrame,
    title=None
):
    """ Plots calibration and prediction data together.

    Plots the given wind field as a vector field flow on the given map.
    Optionally, the calibration data can also plotted as arrows.

    Parameters
    ----------
    map: GeoDataFrame
        The map onto which the wind field is plotted
    calibration_data: WindDataFrame
        The original observation data
    prediction_data: WindDataFrame
        Model prediction data.
    title: string, optional
        The title of the plot, default is None
    """

    # Setup
    fig, ax, x0, x1, xn, y0, y1, yn = setup_plot(map=map, title=title)
    scale = max(x1 - x0, y1 - y0) / 150
    arrow_head_width = max(x1 - x0, y1 - y0) / 100

    # Plot background map as black contour
    map.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)

    # Plot calibration data as red arrows
    calibration_data.apply(lambda row: plot_wind_vector(ax=ax, x=row.x, y=row.y, u=row.u, v=row.v, scale=scale, arrow_head_width=arrow_head_width), axis=1)

    # Plot prediction data as green arrows.
    prediction_data.apply(
        lambda row: plot_wind_vector(
            ax=ax, x=row.x, y=row.y, u=row.u, v=row.v,
            scale=scale,
            arrow_head_width=arrow_head_width,
            color='green'
        ),
        axis=1)


def heatmap(map: GeoDataFrame, calibration_data: GeoDataFrame, windfield: Windfield,
            title=None, plot_calibration_data=False, mask=None, max_wind=10, colormap='Blues'):
    """ Plots the wind speeds as a heatmap.

    Parameters
    ----------
    map: GeoDataFrame
        The map onto which the wind field is plotted
    calibration_data: GeoDataFrame
        The original observation data
    windfield: Windfield
        The windfield to plot
    title: string, optional
        The title of the plot, default is None
    plot_calibration_data: boolean, optional
        Whether to plot the calibration data as arrows, default is False
    mask: GeoSeries (or GeoDataFrame), optional
        If provided, only points within this geometrical shape will be displayed
        on the heatmap, default is None.
    max_wind: int, optional
        The wind speed that will be drawns as the maximum color value, anything
        above this will also be given the maxium color value. Default is 10.
    colormap: string, optional
        The color map to use, see https://matplotlib.org/examples/color/colormaps_reference.html
        Default is 'Blues'
    """

    # Setup
    fig, ax, x0, x1, xn, y0, y1, yn = setup_plot(map=map, title=title)
    scale = max(x1 - x0, y1 - y0) / 150
    arrow_head_width = max(x1 - x0, y1 - y0) / 100

    # Plot background map as black contour
    map.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)

    # Plot calibration data as red arrows
    # if (plot_calibration_data):
    #     calibration_data.apply(lambda row: plot_wind_vector(ax=ax, vector=row.geometry, speed=row.speed, scale=scale, arrow_head_width=arrow_head_width), axis=1)

        # Plot calibration data as red arrows
    if (plot_calibration_data):
            calibration_data.apply(lambda row: plot_wind_vector(ax=ax, x=row.x, y=row.y, u=row.u, v=row.v, scale=scale,
                                                                arrow_head_width=arrow_head_width), axis=1)

    # Plot wind speed as a heat map
    Y, X = np.mgrid[y1:y0:yn*1j, x0:x1:xn*1j]
    # Z = np.zeros_like(X)
    # np.shape(X)
    (m,n) = np.shape(X)
    X = np.reshape(X, (m*n,))
    Y = np.reshape(Y, (m*n,))
    wind = windfield.predict(pd.Series(X), pd.Series(Y))
    speed = np.sqrt(wind.u**2 + wind.v**2)
    speed = np.reshape(speed.values, (m,n))
    if mask is not None:
        zero_speed = np.zeros((m,n))
        points = [Point(x,y) for (x,y) in zip(X,Y)]
        f_points = [mask.contains(p).any() for p in points]
        f_points = np.reshape(f_points, (m,n))
        speed = np.where(f_points, speed, zero_speed)
    # for i in range(xn):
    #     for j in range(yn):
    #         x = X[j, i]
    #         y = Y[j, i]
    #         wind = windfield.get_wind(x, y)
    #
    #         if (mask is None):
    #             Z[j, i] = wind.speed
    #         else:
    #             xp = x + (x1 - x0) / xn
    #             yp = y + (y1 - y0) / yn
    #             if (mask.contains(Point(x, y)).any() or mask.contains(Point(xp, y)).any() or mask.contains(Point(x, yp)).any() or mask.contains(Point(xp, yp)).any()):
    #                 Z[j, i] = wind.speed
    #             else:
    #                 Z[j, i] = 0
    plt.imshow(speed, cmap=colormap, interpolation='bilinear',
               origin='upper', extent=[x0, x1, y0, y1],
               vmin=0, vmax=max_wind)
    # plt.imshow(Z, cmap=colormap, interpolation='bilinear',
    #            origin='upper', extent=[x0, x1, y0, y1],
    #            vmin=0, vmax=max_wind)


def scalar_field(map: GeoDataFrame, field, title=None, mask=None, default=0.0, colormap='Blues', nx = 100, ny = 100):
    """ Plots the wind speeds as a heatmap.

    Parameters
    ----------
    map: GeoDataFrame
        The map onto which the wind field is plotted
    field: Function taking in x,y returning value s(x,y)
    title: string, optional
        The title of the plot, default is None
    plot_calibration_data: boolean, optional
        Whether to plot the calibration data as arrows, default is False
    mask: GeoSeries (or GeoDataFrame), optional
        If provided, only points within this geometrical shape will be displayed
        on the heatmap, default is None.
    max_wind: int, optional
        The wind speed that will be drawns as the maximum color value, anything
        above this will also be given the maxium color value. Default is 10.
    colormap: string, optional
        The color map to use, see https://matplotlib.org/examples/color/colormaps_reference.html
        Default is 'Blues'
    """
    # Setup
    fig, ax, x0, x1, xn, y0, y1, yn = setup_plot(map=map, title=title)

    # Plot background map as black contour
    map.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)

    # Plot scalar field as heat map
    X, Y = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))

    def masked_field(x, y):
        if mask is None or mask.contains(Point(x, y)).any():
            return field(x, y)
        else:
            return default

    vfield = np.vectorize(masked_field)

    Z = vfield(X, Y)
    plt.pcolormesh(X, Y, Z, cmap=colormap, shading='gouraud')
    return fig


def render_figure(fig=None, to_file='figure.pdf', save=False):
    """Save Matplotlib figure on disk if `save` is `True`, otherwise show it.

    If path `to_file` contains directories and they do not exists, then create
    the directories.

    Parameters
    ----------
    fig : handle (optional, default None)
        Handle to the figure to save. If `None`, then the current matplotlib
        figure is used.
    to_file : str
        Filename with or without directories. The extension of the filename
        determines the image format used.
    save : bool (optional, default False)
        Flag showing if the figure is saved to disk or plotted on screen.

    """
    if fig is None:
        fig = plt.gcf()

    if save:
        dirname = os.path.dirname(to_file)
        if dirname and not os.path.isdir(dirname):
            os.makedirs(dirname, bbox_inches='tight')

        fig.savefig(to_file)
    else:
        plt.show()


