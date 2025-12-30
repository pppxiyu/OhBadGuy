dir_crime = './data/data_2017-.csv'
dir_thi_polygon = './data/ThiessenPolygons.shp'
crs_polygon = 'EPSG:2240'
crs_point = 'EPSG:2240'

time_interval = '336h'
len_sequence = 12

gnn_k = 2

dir_camera = './data/camera_nationwide/Atlas of Surveillance-20241128.csv'
dir_camera_geocode = './data/camera_nationwide/Atlas of Surveillance-20241128_geocoded.csv'
dir_nation_population = './data/census/ACSDT5Y2022.B01003_2024-11-29T153303/ACSDT5Y2022.B01003-Data.csv'
dir_county = './data/census/tl_2024_us_county/tl_2024_us_county.shp'

dir_roads = './data/MajorRoads.shp'
dir_local_population = './data/populationDist.shp'

dir_city_boundary = './data/census/Municipal Boundaries.json'

dir_cache = './cache/cache_fullModel_20251108'
model_name = 'GConvGRU'

sensor_count = 10
sim_iteration_num = 2
sim_random_seed = 10

sim_pre_located_cam = [
    {'Road': 224, 'Inflow roads': [109, 225, 269], 'Outflow roads': [3, 195], 'EffectRatio': 1},
    {'Road': 94, 'Inflow roads': [109, 163, 179], 'Outflow roads': [168, 175], 'EffectRatio': 1},
    {'Road': 224, 'Inflow roads': [3, 195], 'Outflow roads': [109, 225, 269], 'EffectRatio': 1},
    {'Road': 66, 'Inflow roads': [105, 153, 200], 'Outflow roads': [168, 210], 'EffectRatio': 1},
    {'Road': 255, 'Inflow roads': [98, 199, 200], 'Outflow roads': [84, 105], 'EffectRatio': 1},
    {'Road': 177, 'Inflow roads': [25, 263], 'Outflow roads': [80, 199], 'EffectRatio': 1},
    {'Road': 80, 'Inflow roads': [14, 28], 'Outflow roads': [177, 199], 'EffectRatio': 1},
    {'Road': 28, 'Inflow roads': [3, 157], 'Outflow roads': [14, 80], 'EffectRatio': 1},
    {'Road': 14, 'Inflow roads': [15, 24], 'Outflow roads': [28, 80], 'EffectRatio': 1},
    {'Road': 39, 'Inflow roads': [32, 34, 37], 'Outflow roads': [36, 162], 'EffectRatio': 1},
    {'Road': 40, 'Inflow roads': [34, 41, 42], 'Outflow roads': [35, 265], 'EffectRatio': 1},
    {'Road': 265, 'Inflow roads': [35, 40], 'Outflow roads': [33, 144], 'EffectRatio': 1},
    {'Road': 265, 'Inflow roads': [33, 144], 'Outflow roads': [35, 40], 'EffectRatio': 1},
    {'Road': 120, 'Inflow roads': [242], 'Outflow roads': [33, 198], 'EffectRatio': 1},
    {'Road': 120, 'Inflow roads': [33, 198], 'Outflow roads': [242], 'EffectRatio': 1},
    {'Road': 119, 'Inflow roads': [198, 273, 275], 'Outflow roads': [241, 242], 'EffectRatio': 1},
    {'Road': 242, 'Inflow roads': [119, 241], 'Outflow roads': [120], 'EffectRatio': 1},
    {'Road': 119, 'Inflow roads': [241, 242], 'Outflow roads': [198, 273, 275], 'EffectRatio': 1}
]

mapbox_key = 'pk.eyJ1IjoicHhpeW9oIiwiYSI6ImNsMGoxa3h1bzA4ZHQzaW41NWd6dm16am0ifQ.QywfLC6Ut-EhSZLt7nirqQ'

dir_figs = './papers/figs'