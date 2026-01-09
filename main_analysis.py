from config import *
import utils.data as dd
from utils import val
import utils.val as val
import utils.vis as vis

analysis_name = 'validation'

if analysis_name == 'nation_dist':
    # nationwide camera distribution
    gdf_pop = dd.match_census_county(dd.read_population_acs(dir_nation_population), dd.read_county(dir_county))
    gdf_pop = dd.get_top_counties(gdf_pop)
    dd.geocode_camera(dir_camera, dir_camera_geocode)
    gdf_cam = dd.read_camera(dir_camera_geocode, gdf_pop.crs)
    gdf_pop = gdf_pop.merge(
        gdf_pop.sjoin(gdf_cam, how='left')[['geo_id', 'lat']].groupby('geo_id').count().rename(
            columns={'lat': 'cam_count'}),
        on='geo_id', how='left'
    )
    gdf_pop['cam_per_pop'] = gdf_pop['cam_count'] / gdf_pop['pop']
    print(
        f'Top {len(gdf_pop[gdf_pop["if_top"] == True])} Counties with half population has'
        f' {gdf_pop[gdf_pop["if_top"] == True]["cam_count"].sum()} cameras, '
        f'{gdf_pop[gdf_pop["if_top"] == True]["cam_per_pop"].mean() * 100000} cameras per 100000 pepople. '
        f'\nThe rest of {len(gdf_pop[gdf_pop["if_top"] == False])} '
        f'counties has {gdf_pop[gdf_pop["if_top"] == False]["cam_count"].sum()} cameras, '
        f'{gdf_pop[gdf_pop["if_top"] == False]["cam_per_pop"].mean() * 100000} cameras per 100000 pepople.'
    )
    vis.map_bar_cam_us(gdf_pop, gdf_cam, mode='map', dir_save=dir_figs)
    vis.map_bar_cam_us(gdf_pop, gdf_cam, mode='bar', dir_save=dir_figs)

if analysis_name == 'city_map':
    crime_data = dd.CrimeData()
    crime_data.import_thi_polygon(dir_thi_polygon, crs_polygon)
    crime_data.import_crime(dir_crime, crs_point)

    road_data = dd.RoadData()
    road_data.read_road_shp(dir_roads)
    road_data.connect_line()
    road_data.convert_roads_2_network()

    vis.map_city_roads_polygon_crime(
        dir_city_boundary, roads=road_data.road_lines.copy(), polygons=crime_data.polygon.copy(), # type: ignore
        crimes=crime_data.crime.copy(), # type: ignore
    )
    vis.map_city_roads_polygon_crime(
        dir_city_boundary, roads=road_data.road_lines.copy(), network=road_data.road_network.copy() # type: ignore
    )

if analysis_name == 'validation':
    import geopandas as gpd
    from shapely import Point
    import statsmodels.api as sm
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta
    from statsmodels.tsa.arima.model import ARIMA
    from scipy import stats
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.regression.linear_model import OLS
    from statsmodels.stats.stattools import durbin_watson

    space_filter_r = 1640  # None,  100m: 328; 500m: 1640; 250m:820
    time_filter_r = 16  # week before and after the implementation point
    resample = 'D'
    effect_date = "2023-05-15" # the implementation data is "2023-04-24"

    ## data of the past one year
    val_data = val.import_val_data(dir_val_data)
    
    # filter using analysis zone
    if space_filter_r is not None:
        plans = val.import_history_plans(dir_implemented_plans)
        plans = plans[~plans['date'].isin(['20231010_static_for_months', 'test'])]
        buffer_zone, df_buffers = val.create_buffer_cams(plans, space_filter_r,)

        # vis:
        # df_buffers = df_buffers.to_crs(epsg=3857)
        # alphas = df_buffers['count'] / df_buffers['count'].max()
        # ax = df_buffers.plot(figsize=(10, 10), alpha=alphas, edgecolor="k")
        # cx.add_basemap(ax, source=cx.providers.Stamen.TonerLite)

        val_data = gpd.GeoDataFrame(
            val_data, geometry = [Point(xy) for xy in zip(val_data['GeoX'], val_data['GeoY'])], crs='2240'
        )
        val_data = val_data[val_data['geometry'].within(buffer_zone)]

    if time_filter_r is not None:
        # filter time range 
        implement_date = pd.to_datetime(effect_date)
        start_date = implement_date - timedelta(weeks=time_filter_r)
        end_date = implement_date + timedelta(weeks=time_filter_r)
        val_data = val_data[
            # (val_data.index >= start_date) 
            # & 
            (val_data.index <= end_date)
        ]

    # vis: temporal line
    before = val_data[val_data.index <= effect_date]
    after = val_data[val_data.index >= effect_date]
    val.line_crime_count(before, after, 'before', 'after', resample=resample, use_moving_avg=True, window=10)

    # formatting 4 reg
    df_crime_sampled = val_data.resample(resample).count()
    df_crime_sampled = df_crime_sampled[['geometry']].rename(columns = {'geometry': 'y'})

    # add time var
    df_crime_sampled['time'] = range(len(df_crime_sampled))

    # add intervention var
    df_crime_sampled['intervention'] = 0
    df_crime_sampled.loc[effect_date:, 'intervention'] = 1

    # add post intervention time var
    df_crime_sampled['time_after_intervention'] = np.where(
        df_crime_sampled['intervention'] == 1,
        df_crime_sampled['time'] - df_crime_sampled[df_crime_sampled['intervention'] == 1]['time'].min(),
        0
    )


    # Use ARIMAX (ARIMA with external regressors)
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    # Try AR(1) first
    model_ar = SARIMAX(df_crime_sampled['y'], 
                    exog=df_crime_sampled[['time', 'intervention', 'time_after_intervention']],
                    order=(1, 0, 0)).fit()

    print(model_ar.summary())












    # regression
    df_crime_sampled_before = df_crime_sampled[df_crime_sampled['series'] == 1].copy()
    df_crime_sampled_after = df_crime_sampled[df_crime_sampled['series'] == 2].copy()

    X = sm.add_constant(df_crime_sampled_before[['time']])
    model = sm.OLS(df_crime_sampled_before['count'], X).fit()
    print(model.summary())

    X = sm.add_constant(df_crime_sampled_after[['time']])
    model = sm.OLS(df_crime_sampled_after['count'], X).fit()
    print(model.summary())







    
print('Done.')
