'''
Helper functions to analyse data on ey-data-challenge
'''
import numpy as np
import pandas as pd
import seaborn as sns
import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, LineString

center = {
    'x_min': 3750901.5068, 'y_min': -19268905.6133,
    'x_max': 3770901.5068, 'y_max': -19208905.6133
}

center['x_middle'] = center['x_min'] + (center['x_max'] - center['x_min'])/2
center['y_middle'] = center['y_min'] + (center['y_max'] - center['y_min'])/2

center['left_border']  = LineString([(center['x_min'], center['y_min']), (center['x_min'], center['y_max'])])
center['right_border'] = LineString([(center['x_max'], center['y_min']), (center['x_max'], center['y_max'])])

center['lower_border']  = LineString([(center['x_min'], center['y_min']), (center['x_max'], center['y_min'])])
center['upper_border']  = LineString([(center['x_min'], center['y_max']), (center['x_max'], center['y_max'])])

center_polygon = Polygon([(center['x_min'], center['y_min']), (center['x_min'], center['y_max']),
                          (center['x_max'], center['y_max']), (center['x_max'], center['y_min'])])

center_polygon_row = pd.DataFrame({ 'geometry': center_polygon }, index=[0])

def entry_to_center_angles(row):
    """
    Calculates the angle between a trajectory's entry and Atlanta's center vertexes.
    
    Gives the angle between two lines: 1) previous trajectory to center vertexes, and 2) previous trajectory to current.
    Computes such angle for all vertexes throughout the city center - and also for its center point. Line 2) is composed of
    both trajectories' entry coordinates.
    
    Parameters
    ----------
    row : pandas.Series
        Row of a DataFrame over which to calculate these angles.
        
    Returns
    -------
    row : pandas.Series
        Input row after resulting angles assignment.
    """
    if np.isnan(row['last_x_entry']) or np.isnan(row['last_y_entry']):
        row[[
            'cc_middle_angle', 'cc_xmin_ymin_angle', 'cc_xmax_ymin_angle', 'cc_xmin_ymax_angle', 'cc_xmax_ymax_angle'
        ]] = np.nan
        
        return row
    
    last_to_current_entry = LineString([(row['last_x_entry'], row['last_y_entry']), (row['x_entry'], row['y_entry'])])
    
    to_center_lines = {
        'middle': LineString([(row['last_x_entry'], row['last_y_entry']), (center['x_middle'], center['y_middle'])]),
        'xmin_ymin': LineString([(row['last_x_entry'], row['last_y_entry']), (center['x_min'], center['y_min'])]),
        'xmin_ymax': LineString([(row['last_x_entry'], row['last_y_entry']), (center['x_min'], center['y_max'])]),
        'xmax_ymin': LineString([(row['last_x_entry'], row['last_y_entry']), (center['x_max'], center['y_min'])]),
        'xmax_ymax': LineString([(row['last_x_entry'], row['last_y_entry']), (center['x_max'], center['y_max'])])
    }
    
    for vertex in ['xmin_ymin', 'xmax_ymin', 'xmin_ymax', 'xmax_ymax', 'middle']:
        row[f'cc_{vertex}_angle'] = angle_between(last_to_current_entry, to_center_lines[vertex])
    
    return row
    
def angle_between(line1, line2):
    """
    The angle between two lines.
    
    Valid angle calculation for any two lines, even if any of them or both are vertical.
    
    Parameters
    ----------
    line1 : shapely.LineString
        First line for comparison.
    line2 : shapely.LineString
        Second line for comparison.
    
    Returns
    -------
    float
        The resulting angle in rads.
    """
    coords_1 = line1.coords
    coords_2 = line2.coords
    
    line1_vertical = (coords_1[1][0] - coords_1[0][0]) == 0.0
    line2_vertical = (coords_2[1][0] - coords_2[0][0]) == 0.0
    
    # Vertical lines have undefined slope, but we know their angle in rads is = 90° * π/180
    if line1_vertical and line2_vertical:
        # Perpendicular vertical lines
        return 0.0
    if line1_vertical or line2_vertical:
        # 90° - angle of non-vertical line
        non_vertical_line = line2 if line1_vertical else line1
        return abs((90.0 * np.pi / 180.0) - np.arctan(slope(non_vertical_line)))
    
    m1 = slope(line1)
    m2 = slope(line2)
    
    return abs(np.arctan((m1 - m2)/(1 + m1*m2)))

def slope(line):
    """
    Cartesian line slope, classically defined as Δy/Δx.
    
    Parameters
    ----------
    line: shapely.LineString
        The line for which to calculate slope.
    """
    x0 = line.coords[0][0]
    y0 = line.coords[0][1]
    x1 = line.coords[1][0]
    y1 = line.coords[1][1]
    return (y1 - y0) / (x1 - x0)

def geoplot(sample_df, figsize=(25, 27), ax=None, start='15:00:00', end='16:00:00'):
    """
    Plots a DataFrame's entry-to-exit trajectories in a cartesian plane, along to the city center polygon.
    
    Creates shapely.LineString's representing entry-to-exit cartesian trajectories throughout the DataFrame,
    then plotting those within a predefined time frame. Will always plot the Atlanta's center polygon alongside.
    
    Parameters
    ----------
    sample_df : pandas.DataFrame
        The sample DataFrame from which to plot trajectories.
    figsize : (int, int)
        Matplotlib figsize for the plot.
    ax : matplotlib.axes._subplots.AxesSubplot
        Matplotlib ax into which to plot.
    start : str
        Start of trajectory time frame sample. Formatted as HH:mm:ss.
    end : str
        End of trajectory time frame sample. Formatted as HH:mm:ss.
    """
    # Time-wise sampling
    starttime = pd.to_timedelta(start)
    endtime = pd.to_timedelta(end)
    
    samples = sample_df[(sample_df.time_entry >= starttime) & (sample_df.time_exit <= endtime)]
    samples['geometry'] = samples.apply(
        lambda row: LineString([ (row['x_entry'], row['y_entry']), (row['x_exit'], row['y_exit']) ]),
        axis = 1
    )
    
    # Addition of Atlanta's center polygon
    samples_plus_center = pd.concat([samples, center_polygon_row], sort=False)
    
    # Geo plot
    geodf = gpd.GeoDataFrame(samples_plus_center)
    colors = cm.rainbow(np.linspace(0, 1, len(geodf)))
    
    if ax is None:
        fig = plt.figure(1, dpi=90, figsize=figsize)
        ax = fig.add_subplot(121)

    geodf.plot(ax=ax, color=colors)

def is_inside_city(x, y):
    if (center['x_min'] <= x <= center['x_max']) and (center['y_min'] <= y <= center['y_max']):
        return 1
    else:
        return 0

def euclidian_distance(x_one, y_one, x_two, y_two):
    """Distance as defined by the Euclidian formula for the ((x1, y1), (x2, y2)) case."""
    return np.sqrt(np.power((x_one-x_two), 2) + np.power((y_one-y_two), 2))

def dist_to_center(condition, middle_prop, entry, df):
    dist = abs( center[middle_prop] - df[condition][entry].values )
    return dist / dist.max()

def center_permanency(row):
    """
    Computes the percentage of a trajectory's stay in Atlanta's center.
    
    Having an entry-to-exit trajectory with length n, its center permanency represents
    the percentage of n that happens inside Atlanta's center polygon, in a [0,1] interval.
    Trajectories starting and ending within the center will then have 
    """
    if any(np.isnan(row[col]) for col in ['x_exit', 'y_exit']):
        return np.nan
    
    line = LineString([(row['x_entry'], row['y_entry']), (row['x_exit'], row['y_exit'])])
    
    if not line.intersects(center_polygon):
            return 0
    
    if line.length == 0:
        # Avoids divisions by 0 in 'point' trajectories
        return 1
    
    return line.intersection(center_polygon).length / line.length

def distplot(df, index, figsize=(12, 8), groupby='hash', label='', ax=None):
    grouped = df.groupby(groupby)[index]

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    
    sns.set(style="white", palette="muted", color_codes=True)
    sns.distplot(grouped.mean().values, ax=ax, label=label)

def plot_avg_distance(avg_list, figsize, title='Average Distance Traveled', labels=[], ax=None):
    if ax is None:
        ax = plt.subplot(len(avg_list), 1, 1)
    
    ax.set_ylabel(title)

    if labels is not None and len(labels) == len(avg_list):
        for avg, label in zip(avg_list, labels):
            avg.plot(ax=ax, figsize=figsize, label=label)
    else:
        for avg in avg_list:
            avg.plot(ax=ax, figsize=figsize)
        
    plt.legend()