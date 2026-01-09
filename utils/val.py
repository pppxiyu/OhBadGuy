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


def line_crime_count(df1, df2, df1Label, df2Label, resample, typeFilter=None, 
                     use_moving_avg=False, window=7):
    """
    Plot crime counts over time with average count reference lines.
    
    Parameters:
    -----------
    df1, df2 : pd.DataFrame
        DataFrames with datetime index and 'Type' column
    df1Label, df2Label : str
        Labels for the two datasets
    resample : str
        Resampling frequency (e.g., 'D', 'W', 'M')
    typeFilter : str, optional
        Filter for specific crime type (e.g., 'Assault Offenses', 'Larceny')
    use_moving_avg : bool, default False
        If True, plot moving average instead of raw counts
    window : int, default 7
        Window size for moving average (only used if use_moving_avg=True)
        
    Common crime types:
        - Assault Offenses
        - Larceny
        - Vandalism
        - Burglary
        - Theft of Motor Vehicles
    """
    # Filter by crime type if specified
    if typeFilter is not None:
        df1 = df1[df1.Type == typeFilter]
        df2 = df2[df2.Type == typeFilter]
    
    # Resample and count
    crimeCount_df1 = df1.resample(resample).count()
    crimeCount_df2 = df2.resample(resample).count()
    
    # Get the first column for counting
    count_col1 = crimeCount_df1.columns[0]
    count_col2 = crimeCount_df2.columns[0]
    
    # Prepare data for plotting (raw or moving average)
    if use_moving_avg:
        plot_data1 = crimeCount_df1[count_col1].rolling(window=window, center=True).mean()
        plot_data2 = crimeCount_df2[count_col2].rolling(window=window, center=True).mean()
        curve_label_suffix = f' (MA-{window})'
    else:
        plot_data1 = crimeCount_df1[count_col1]
        plot_data2 = crimeCount_df2[count_col2]
        curve_label_suffix = ''
    
    # Calculate averages (always from original data)
    avg_df1 = crimeCount_df1[count_col1].mean()
    avg_df2 = crimeCount_df2[count_col2].mean()
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Plot crime count curves (raw or moving average)
    plt.plot(crimeCount_df1.index, plot_data1, 
             label=f'{df1Label}{curve_label_suffix}', linewidth=2, alpha=0.8)
    plt.plot(crimeCount_df2.index, plot_data2, 
             label=f'{df2Label}{curve_label_suffix}', linewidth=2, alpha=0.8)
    
    # Plot average lines (based on original data)
    plt.axhline(y=avg_df1, color='C0', linestyle='--', linewidth=1.5, 
                label=f'{df1Label} Avg: {avg_df1:.2f}', alpha=0.7)
    plt.axhline(y=avg_df2, color='C1', linestyle='--', linewidth=1.5, 
                label=f'{df2Label} Avg: {avg_df2:.2f}', alpha=0.7)
    
    # Labels and formatting
    plt.xlabel('Date', fontsize=12)
    ylabel = f'Count / {resample}'
    if use_moving_avg:
        ylabel += f' (Moving Avg, window={window})'
    plt.ylabel(ylabel, fontsize=12)
    
    # Title with crime type if filtered
    title = 'Crime Count Over Time'
    if typeFilter is not None:
        title += f': {typeFilter}'
    else:
        title += ': All Types'
    if use_moving_avg:
        title += f' [Moving Average]'
    plt.title(title, fontsize=14, pad=15)
    
    plt.legend(loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n{'='*50}")
    print(f"Summary Statistics:")
    print(f"{'='*50}")
    print(f"{df1Label}:")
    print(f"  Average: {avg_df1:.2f}")
    print(f"  Min: {crimeCount_df1[count_col1].min():.0f}")
    print(f"  Max: {crimeCount_df1[count_col1].max():.0f}")
    if use_moving_avg:
        print(f"  MA Min: {plot_data1.min():.2f}")
        print(f"  MA Max: {plot_data1.max():.2f}")
    
    print(f"\n{df2Label}:")
    print(f"  Average: {avg_df2:.2f}")
    print(f"  Min: {crimeCount_df2[count_col2].min():.0f}")
    print(f"  Max: {crimeCount_df2[count_col2].max():.0f}")
    if use_moving_avg:
        print(f"  MA Min: {plot_data2.min():.2f}")
        print(f"  MA Max: {plot_data2.max():.2f}")
    print(f"{'='*50}\n")


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
