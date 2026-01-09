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
        start_date = implement_date - timedelta(weeks= 48 - time_filter_r)
        end_date = implement_date + timedelta(weeks=time_filter_r)
        val_data = val_data[
            # (val_data.index >= start_date) 
            # & 
            (val_data.index <= end_date)
        ]

    # vis: temporal line
    before = val_data[val_data.index <= effect_date]
    after = val_data[val_data.index >= effect_date]
    vis.line_crime_count(before, after, 'before', 'after', resample=resample, use_moving_avg=True, window=10)

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



    import matplotlib.pyplot as plt
    import numpy as np

    # Get fitted values and residuals
    df_crime_sampled['fitted'] = result.fittedvalues
    df_crime_sampled['residuals'] = result.resid

    # Intervention point
    intervention_idx = df_crime_sampled['intervention'].idxmax()
    intervention_time = df_crime_sampled.loc[intervention_idx, 'time']

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_crime_sampled['time'], df_crime_sampled['y'], 'o', alpha=0.5, label='Actual', markersize=3)
    ax.plot(df_crime_sampled['time'], df_crime_sampled['fitted'], '-', color='red', label='Fitted', linewidth=2)
    ax.axvline(intervention_time, color='green', linestyle='--', linewidth=2, label='Intervention')
    ax.set_xlabel('Time')
    ax.set_ylabel('Crime Count')
    ax.set_title('Actual vs Fitted Values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()




    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    from statsmodels.stats.diagnostic import het_white

    # Get fitted values and residuals
    df_crime_sampled['fitted'] = result.fittedvalues
    df_crime_sampled['residuals'] = result.resid

    # Intervention point
    intervention_idx = df_crime_sampled['intervention'].idxmax()
    intervention_time = df_crime_sampled.loc[intervention_idx, 'time']

    # Statistical tests for residuals
    resid = df_crime_sampled['residuals'].values

    # Zero Mean test (t-test)
    t_stat, p_value_mean = stats.ttest_1samp(resid, 0)
    mean_resid = resid.mean()
    std_resid = resid.std()

    # Homoscedasticity tests
    # White test
    try:
        white_stat, white_pvalue, _, _ = het_white(resid, result.model.exog)
    except:
        white_stat, white_pvalue = np.nan, np.nan

    # Levene's test (pre vs post intervention)
    pre_resid = df_crime_sampled[df_crime_sampled['intervention'] == 0]['residuals']
    post_resid = df_crime_sampled[df_crime_sampled['intervention'] == 1]['residuals']
    levene_stat, levene_pvalue = stats.levene(pre_resid, post_resid)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_crime_sampled['time'], df_crime_sampled['residuals'], 'o-', alpha=0.6, markersize=3, color='steelblue')

    # Add mean line
    ax.axhline(mean_resid, color='red', linestyle='-', linewidth=2, label=f'Mean = {mean_resid:.4f}')
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Zero')

    # Add ±2 std bands
    ax.axhline(mean_resid + 2*std_resid, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='±2σ')
    ax.axhline(mean_resid - 2*std_resid, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)

    # Intervention line
    ax.axvline(intervention_time, color='green', linestyle='--', linewidth=2, label='Intervention')

    # Add test results as text box
    if not np.isnan(white_pvalue):
        textstr = '\n'.join([
            'Statistical Tests:',
            f'Zero Mean (t-test):',
            f'  p = {p_value_mean:.4f}',
            f'  {"✓ Pass" if p_value_mean > 0.05 else "✗ Fail"} (α=0.05)',
            '',
            f'Homoscedasticity:',
            f'  White test: p = {white_pvalue:.4f}',
            f'  {"✓ Pass" if white_pvalue > 0.05 else "✗ Fail"}',
            f'  Levene (pre/post): p = {levene_pvalue:.4f}',
            f'  {"✓ Pass" if levene_pvalue > 0.05 else "✗ Fail"}'
        ])
    else:
        textstr = '\n'.join([
            'Statistical Tests:',
            f'Zero Mean (t-test):',
            f'  p = {p_value_mean:.4f}',
            f'  {"✓ Pass" if p_value_mean > 0.05 else "✗ Fail"} (α=0.05)',
            '',
            f'Homoscedasticity:',
            f'  Levene (pre/post): p = {levene_pvalue:.4f}',
            f'  {"✓ Pass" if levene_pvalue > 0.05 else "✗ Fail"}'
        ])

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=props, family='monospace')

    ax.set_xlabel('Time')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals Over Time with Diagnostics')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Print detailed test results
    print("\n" + "="*50)
    print("RESIDUAL DIAGNOSTIC TESTS")
    print("="*50)
    print(f"\n1. Zero Mean Test (One-sample t-test)")
    print(f"   H0: mean = 0")
    print(f"   Sample mean: {mean_resid:.6f}")
    print(f"   t-statistic: {t_stat:.4f}")
    print(f"   p-value: {p_value_mean:.4f}")
    print(f"   Result: {'PASS - Mean not significantly different from 0' if p_value_mean > 0.05 else 'FAIL - Mean significantly different from 0'}")

    print(f"\n2. Homoscedasticity Tests")
    if not np.isnan(white_pvalue):
        print(f"   a) White Test")
        print(f"      H0: Homoscedastic (constant variance)")
        print(f"      LM-statistic: {white_stat:.4f}")
        print(f"      p-value: {white_pvalue:.4f}")
        print(f"      Result: {'PASS - Constant variance' if white_pvalue > 0.05 else 'FAIL - Heteroscedasticity detected'}")
        print(f"\n   b) Levene Test (Pre vs Post Intervention)")
    else:
        print(f"   a) Levene Test (Pre vs Post Intervention)")

    print(f"      H0: Equal variances")
    print(f"      Statistic: {levene_stat:.4f}")
    print(f"      p-value: {levene_pvalue:.4f}")
    print(f"      Std (pre): {pre_resid.std():.4f}, Std (post): {post_resid.std():.4f}")
    print(f"      Result: {'PASS - Equal variances' if levene_pvalue > 0.05 else 'FAIL - Unequal variances'}")
    print("="*50)





    import matplotlib.pyplot as plt
    import numpy as np

    # Get fitted values and residuals
    df_crime_sampled['fitted'] = result.fittedvalues
    df_crime_sampled['residuals'] = result.resid

    # Calculate statistics
    resid = df_crime_sampled['residuals'].values
    mean_resid = resid.mean()
    std_resid = resid.std()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df_crime_sampled['residuals'], bins=30, density=True, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Residuals')
    ax.set_ylabel('Density')
    ax.set_title('Residual Distribution')

    # Add normal curve overlay
    x = np.linspace(df_crime_sampled['residuals'].min(), df_crime_sampled['residuals'].max(), 100)
    ax.plot(x, 1/(std_resid * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean_resid) / std_resid)**2), 
            'r-', linewidth=2, label='Normal')
    ax.axvline(mean_resid, color='red', linestyle='--', linewidth=1.5, label=f'Mean={mean_resid:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



    
print('Done.')
