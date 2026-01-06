from config import *
import utils.data as dd
from utils import visualization as vis

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
    vis.map_bar_cam_us(gdf_pop, gdf_cam, mode='map', dir_save=dir_fig_save)
    vis.map_bar_cam_us(gdf_pop, gdf_cam, mode='bar', dir_save=dir_fig_save)

if analysis_name == 'city_map':
    crime_data = dd.CrimeData()
    crime_data.import_thi_polygon(dir_thi_polygon, crs_polygon)
    crime_data.import_crime(dir_crime, crs_point)

    road_data = dd.RoadData()
    road_data.read_road_shp(dir_roads)
    road_data.connect_line()
    road_data.convert_roads_2_network()

    vis.map_city_roads_polygon_crime(
        dir_city_boundary, roads=road_data.road_lines.copy(), polygons=crime_data.polygon.copy(),
        crimes=crime_data.crime.copy(),
    )
    vis.map_city_roads_polygon_crime(
        dir_city_boundary, roads=road_data.road_lines.copy(), network=road_data.road_network.copy()
    )

if analysis_name == 'validation':
    # # one data pull of the past one year
    onePull = processRawData('./data/Aug 28, 2022-Aug 28, 2023.xlsx')
    onePull = onePull.set_index('TimeOccur')
    before = onePull[(onePull.index > '2022-12-24 00:00:00') & (onePull.index < '2023-04-24 00:00:00')]
    after = onePull[(onePull.index > '2023-04-24 00:00:00') & (onePull.index < '2023-08-24 00:00:00')]

    before.resample('D').count().mean()
    after.resample('D').count().mean()

    crimeCountBefore = before.resample('D').count()['GeoX'].values
    crimeCountAfter = after.resample('D').count()['GeoX'].values
    t_stat, p_value = ttest_ind(crimeCountAfter, crimeCountBefore, alternative = 'less')

    alpha = 0.10
    if p_value < alpha:
        print(f"The mean of array1 is statistically lower than the mean of array2 (p-value = {p_value:.5f})")
    else:
        print(f"There's no evidence to suggest that the mean of array1 is lower than the mean of array2 (p-value = {p_value:.5f})")

    # vis
    crimeCount_line(before, after, 'before', 'after', typeFilter = None)

    onePull.resample('D').count()

    ## One single data pull with narrowed area
    plans = getPlans('./plans')
    buffer_zone, buffers = create_buffer_cams(plans, 820, 0)  # 100m: 328; 500m: 1640; 250m:820

    buffers = buffers.to_crs(epsg=3857)
    alphas = buffers['count'] / buffers['count'].max()
    ax = buffers.plot(figsize=(10, 10), alpha=alphas, edgecolor="k")
    # cx.add_basemap(ax, source=cx.providers.Stamen.TonerLite)

    # # one data pull of the past one year
    onePull = processRawData('./data/Aug 28, 2022-Aug 28, 2023.xlsx')
    onePull = onePull.set_index('TimeOccur')
    onePull = gpd.GeoDataFrame(onePull, geometry = [Point(xy) for xy in zip(onePull['GeoX'], onePull['GeoY'])])
    onePull = onePull.set_crs(epsg = 2240)

    # filter data using buffer zones
    onePull_filter = onePull[onePull['geometry'].within(buffer_zone)]

    before = onePull_filter [(onePull_filter.index > '2022-12-24 00:00:00') & (onePull_filter.index < '2023-04-24 00:00:00')]
    after = onePull_filter [(onePull_filter.index > '2023-04-23 00:00:00') & (onePull_filter.index < '2023-08-24 00:00:00')]

    before.resample('D').count().mean()
    after.resample('D').count().mean()

    crimeCountBefore = before.resample('D').count()['GeoX'].values
    crimeCountAfter = after.resample('D').count()['GeoX'].values
    t_stat, p_value = ttest_ind(crimeCountAfter, crimeCountBefore, alternative = 'less')

    alpha = 0.10
    if p_value < alpha:
        print(f"The mean of array1 is statistically lower than the mean of array2 (p-value = {p_value:.5f})")
    else:
        print(f"There's no evidence to suggest that the mean of array1 is lower than the mean of array2 (p-value = {p_value:.5f})")

    ## check if the increase trend is lower in cam buffer
    crime_week = onePull_filter.resample('D').count().iloc[1: -1, :]
    crime_week = crime_week[['geometry']].rename(columns = {'geometry': 'count'})
    crime_week['series'] = 0
    crime_week.loc[:"2023-04-24", 'series'] = 1
    crime_week.loc["2023-04-24":, 'series'] = 2

    implement_loc = crime_week.index.get_loc("2023-04-24")
    crime_week = crime_week.iloc[implement_loc - 7 * 4: implement_loc + 7 * 4, :]

    crime_week['time'] = range(len(crime_week))
    crime_week_before = crime_week[crime_week['series'] == 1].copy()
    crime_week_after = crime_week[crime_week['series'] == 2].copy()

    X = sm.add_constant(crime_week_before[['time']])
    model = sm.OLS(crime_week_before['count'], X).fit()
    print(model.summary())

    new_X = sm.add_constant(crime_week_after[['time']])
    model.predict(new_X).mean()

    crime_week_after['count'].mean()

    X = sm.add_constant(crime_week_after[['time']])
    model = sm.OLS(crime_week_after['count'], X).fit()
    print(model.summary())

    # Generate sample data
    np.random.seed(42)
    time = np.arange(100)
    ts1 = time * 0.5 + np.random.randn(100) * 5
    ts2 = time * 0.8 + np.random.randn(100) * 5

    # Combine the two series into a single DataFrame
    df = pd.DataFrame({
        'Time': np.concatenate([time, time]),
        'Value': np.concatenate([ts1, ts2]),
        'Series': np.concatenate([np.repeat(1, 100), np.repeat(2, 100)])
    })

    # Create an interaction term
    df['Interaction'] = df['Time'] * df['Series']

    # Fit the model
    X = sm.add_constant(df[['Time', 'Series', 'Interaction']])
    model = sm.OLS(df['Value'], X).fit()

    # Check the coefficient of the interaction term
    print(model.summary())











    
print('Done.')
