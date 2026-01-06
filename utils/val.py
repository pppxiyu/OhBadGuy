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

def getData(dir, startTime, endTime):
    df = pd.read_csv(dir)
    df.TimeOccur = pd.to_datetime(df.TimeOccur)
    df = df.set_index('TimeOccur')
    df = df[(df.index > startTime) & (df.index < endTime)]
    df = gpd.GeoDataFrame(df, geometry = [Point(xy) for xy in zip(df['GeoX'], df['GeoY'])])
    df = df.set_crs(epsg = 2240)
    return df

def crimeCount_line(df1, df2, df1Labe, df2Labe, typeFilter = None):
    """
    most common crime type:
    Assault Offenses, Larceny,
    Vandalism, Burglary
    Theft of Motor Vehicles
    """
    if typeFilter != None:
        df1 = df1[df1.Type == typeFilter]
        df2 = df2[df2.Type == typeFilter]
    crimeCount_df1 = df1.resample('D').count()
    crimeCount_df2 = df2.resample('D').count()

    plt.figure(figsize = (7.5, 5))
    plt.plot(crimeCount_df1[crimeCount_df1.columns[0]], label = df1Labe)
    plt.plot(crimeCount_df2[crimeCount_df2.columns[0]], label = df2Labe)
    plt.xlabel('Date')
    plt.ylabel('Count / Day')
    plt.legend()
    plt.show()

def processRawData(file):
    df = pd.read_excel(file)
    df = df[['date_occu', 'geox', 'geoy', 'groupa']]
    df = df.rename(columns={'date_occu':'TimeOccur', 'geox':'GeoX', 'geoy':'GeoY', 'groupa':'Type'})
    df['TimeOccur'] = pd.to_datetime(df['TimeOccur'],format='%Y/%m/%d %H:%M:%S')
    df = df.sort_values(by = ["TimeOccur"], ascending = False)
    # df['TimeOccur'] = pd.to_datetime(df['TimeOccur'],format='%m/%d/%Y %H:%M:%S').dt.strftime('%m/%d/%Y %H:%M:%S')
    # df.drop(df[df['GeoX'] <= 0.0].index, inplace=True)
    # df.drop(df[df['GeoY'] <= 0.0].index, inplace=True)
    return df


def getPlans(dir):
    plans = []
    for filename in os.listdir(dir):
        if filename.endswith(".html"):
            # if filename != 'cameraPlacement20230717.html':
            with open(dir + '/' + filename, 'r') as file:
                html_content = file.read()
            stringSelect = html_content[html_content.index('"type":"choroplethmapbox"},{"customdata"') : html_content.index('"type":"choroplethmapbox"},{"customdata"') + 1000]
            stringSelect = stringSelect[stringSelect.index('"customdata"'): stringSelect.index(',"hovertemplate"')]
            stringSelect = stringSelect[stringSelect.index('[['): ]
            plans.append(ast.literal_eval(stringSelect))
    return plans


def getCamFrequency(plans):
    cams = [cam[0:2] for week in plans for cam in week]
    df = pd.Series(cams).value_counts().to_frame().reset_index()
    df.columns = ['cam', 'count']
    df['lat'] = df.cam.str[0]
    df['lon'] = df.cam.str[1]
    gdf = gpd.GeoDataFrame(df, geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])])
    gdf = gdf.set_crs(epsg = 4326)
    return gdf


def create_buffer_cams(plans, buffer_dist, threshold_place_count = 0):

    # organize placements with frequency
    camDf = getCamFrequency(plans)
    camDf = camDf.to_crs(epsg = 2240)
    camDf = camDf[camDf['count'] > threshold_place_count]

    # make buffur zone of the placements
    camDf['buffer'] = camDf['geometry'].buffer(buffer_dist)
    camDf = camDf.set_geometry('buffer')

    # merge the buffers
    camDf['dissolve_label'] = 1
    buffer = camDf.dissolve(by='dissolve_label').geometry.iloc[0]

    return buffer, camDf
