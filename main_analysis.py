from config import *
import data as dd
from utils import visualization as vis

analysis_name = None

if analysis_name == 'nation_dist':
    # nationwide camera distribution
    gdf_pop = dd.match_census_county(dd.read_population(dir_population), dd.read_county(dir_county))
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
        f'Top {len(gdf_pop[gdf_pop['if_top'] == True])} Counties with half population has'
        f' {gdf_pop[gdf_pop['if_top'] == True]['cam_count'].sum()} cameras, '
        f'{gdf_pop[gdf_pop['if_top'] == True]['cam_per_pop'].mean() * 100000} cameras per 100000 pepople. '
        f'\nThe rest of {len(gdf_pop[gdf_pop['if_top'] == False])} '
        f'counties has {gdf_pop[gdf_pop['if_top'] == False]['cam_count'].sum()} cameras, '
        f'{gdf_pop[gdf_pop['if_top'] == False]['cam_per_pop'].mean() * 100000} cameras per 100000 pepople.'
    )
    vis.map_bar_cam_us(gdf_pop, gdf_cam, mode='map', dir_save=dir_fig_save)
    vis.map_bar_cam_us(gdf_pop, gdf_cam, mode='bar', dir_save=dir_fig_save)

print()
