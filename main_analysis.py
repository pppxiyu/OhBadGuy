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
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta
    from scipy import stats
    import matplotlib.pyplot as plt
    from statsmodels.tsa.statespace.sarimax import SARIMAX

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

    # # vis: temporal line
    before = val_data[val_data.index <= effect_date]
    after = val_data[val_data.index >= effect_date]
    vis.line_crime_count(
        before, after, 'before', 'after', resample=resample, use_moving_avg=True, window=14,
        save_path=f'{dir_figs}/line_crime_count_before_after.png'
    )

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

    # Use ARIMA with external regressors
    result = SARIMAX(
        df_crime_sampled['y'],
        exog=df_crime_sampled[['time', 'time_after_intervention']],
        order=(4, 0, 0),
    ).fit()

    # View results
    print(result.summary())

    # Test if trend weakened (coefficient on time_after_intervention < 0)
    coef = result.params['time_after_intervention']
    pvalue = result.pvalues['time_after_intervention']
    print(f"\nSlope change coefficient: {coef:.4f}")
    print(f"P-value (one-sided test H1: coef < 0): {pvalue/2 if coef < 0 else 1-pvalue/2:.4f}")

    # # vis: val reg
    # vis.line_crime_count_fitting(df_crime_sampled, result, save_path=f'{dir_figs}/line_crime_count_fitting.png')
    # vis.line_fitting_rediduals(df_crime_sampled, result, save_path=f'{dir_figs}/line_fitting_redisual.png')
    # vis.dist_fitting_residuals(df_crime_sampled, result, save_path=f'{dir_figs}/dist_fitting_redisual.png')


print('Done.')
