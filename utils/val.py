import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind
import os
import ast
import statsmodels.api as sm
import contextily as cx


def import_val_data(file):
    df = pd.read_csv(file)
    df = df[['date_occu', 'geox', 'geoy', 'groupa']]
    df.rename(columns={'date_occu':'TimeOccur', 'geox':'GeoX', 'geoy':'GeoY', 'groupa':'Type'}, inplace=True)
    df['TimeOccur'] = pd.to_datetime(df['TimeOccur'],format='mixed')
    df = df.sort_values(by = ["TimeOccur"], ascending = False)
    df = df.set_index('TimeOccur')
    return df


def import_history_plans(dir):
    """
    Import camera placement plans from HTML files.

    Args:
        dir: Directory containing HTML files in format 'cameraPlacement_YYYYMMDD.html'

    Returns:
        DataFrame with columns:
        - date: Extracted date from filename (YYYYMMDD format)
        - plans: List of camera placements from the HTML file
    """
    data = []
    for filename in os.listdir(dir):
        if filename.endswith(".html") and filename.startswith("cameraPlacement_"):
            # Extract date from filename (remove "cameraPlacement_" prefix and ".html" suffix)
            date = filename.replace("cameraPlacement_", "").replace(".html", "")

            # Read and parse HTML content
            with open(os.path.join(dir, filename), 'r', encoding='utf-8') as file:
                html_content = file.read()

            # Extract plans data from HTML
            stringSelect = html_content[html_content.index('"type":"choroplethmapbox"},{"customdata"') : html_content.index('"type":"choroplethmapbox"},{"customdata"') + 1000]
            stringSelect = stringSelect[stringSelect.index('"customdata"'): stringSelect.index(',"hovertemplate"')]
            stringSelect = stringSelect[stringSelect.index('[['): ]
            plans = ast.literal_eval(stringSelect)

            data.append({'date': date, 'plans': plans})

    # Create DataFrame and sort by date
    df = pd.DataFrame(data)
    df = df.sort_values('date').reset_index(drop=True)

    return df


def calc_plc_frequency(plans):
    from collections import Counter

    # Extract all coordinates from all plans
    all_coords = []
    for plan_list in plans['plans']:
        for location in plan_list:
            # Create a tuple of (lat, lon) for counting
            coord = (location[0], location[1])
            all_coords.append(coord)

    # Count frequencies
    coord_counts = Counter(all_coords)

    # Create new DataFrame
    coord_freq_df = pd.DataFrame([
        {'coordinates': coord, 'frequency': count} 
        for coord, count in coord_counts.items()
    ])

    # Sort by frequency (optional)
    coord_freq_df = coord_freq_df.sort_values('frequency', ascending=False).reset_index(drop=True)

    # create gdf
    coord_freq_df['lat'] = coord_freq_df.coordinates.str[0]
    coord_freq_df['lon'] = coord_freq_df.coordinates.str[1]
    gdf = gpd.GeoDataFrame(coord_freq_df, geometry = [Point(xy) for xy in zip(coord_freq_df['lon'], coord_freq_df['lat'])])
    gdf = gdf.set_crs(epsg = 4326)

    return gdf


def create_buffer_cams(plans, buffer_dist, threshold_count=0):
    # organize placements with frequency
    df_plc = calc_plc_frequency(plans)
    df_plc = df_plc.to_crs(epsg = 2240)
    df_plc = df_plc[df_plc['frequency'] > threshold_count]

    # make buffur zone of the placements
    df_plc['buffer'] = df_plc['geometry'].buffer(buffer_dist)
    df_plc = df_plc.set_geometry('buffer')

    # merge the buffers
    df_plc['dissolve_label'] = 1
    analysis_area = df_plc.dissolve(by='dissolve_label').geometry.iloc[0]

    return analysis_area, df_plc


