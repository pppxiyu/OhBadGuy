from config import *
import utils.data as dd
from utils import val
import utils.val as val
import utils.vis as vis

import geopandas as gpd
from shapely import Point
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

"""
Instructions:
1. Place 20220828-20230828.csv under a folder named data located in the root directory
2. The following code only import and process and raw data. Parameters have been included below.
2. Consider using Prophet to build a baseline
"""


space_filter_r = 1640  # None,  100m: 328; 500m: 1640; 250m:820
time_filter_r = 16  # week (before and) after the implementation point
resample = 'D'
effect_date = "2023-04-24" # the implementation data is "2023-04-24"

## data of the past one year
val_data = val.import_val_data(dir_val_data)

# filter using analysis zone
if space_filter_r is not None:
    plans = val.import_history_plans(dir_implemented_plans)
    plans = plans[~plans['date'].isin(['20231010_static_for_months', 'test'])]
    buffer_zone, df_buffers = val.create_buffer_cams(plans, space_filter_r,)

    # # vis: buffered areas
    # road_data = dd.RoadData()
    # road_data.read_road_shp(dir_roads)
    # vis.map_plc_over_time(
    #     df_buffers, dir_city_boundary, road_data.road_lines.copy(), save_path=f'{dir_figs}/map_plc_over_weeks.png'
    # )

    val_data = gpd.GeoDataFrame(
        val_data, geometry = [Point(xy) for xy in zip(val_data['GeoX'], val_data['GeoY'])], crs='2240'
    )
    val_data = val_data[val_data['geometry'].within(buffer_zone)]

if time_filter_r is not None:
    # filter time range 
    implement_date = pd.to_datetime(effect_date)
    start_date = implement_date - timedelta(weeks= 48 - time_filter_r)
    end_date = implement_date + timedelta(weeks=time_filter_r)
    val_data = val_data[
        # (val_data.index >= start_date) 
        # & 
        (val_data.index <= end_date)
    ]

# # # vis: temporal line
# before = val_data[val_data.index <= effect_date]
# after = val_data[val_data.index >= effect_date]
# vis.line_crime_count(
#     before, after, 'before', 'after', resample=resample, use_moving_avg=True, window=14,
#     save_path=f'{dir_figs}/line_crime_count_before_after.png'
# )

print('Done.')
