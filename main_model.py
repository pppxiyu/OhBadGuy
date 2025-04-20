from config import *
from utils import data as dd
from utils import evaluation as ev
import model as mo

# crime data
crime_data = dd.CrimeData()
crime_data.import_thi_polygon(dir_thi_polygon, crs_polygon)
crime_data.import_crime(dir_crime, crs_point)
crime_resampled = crime_data.resample_crime(time_interval)
train_x, train_y, val_x, val_y, test_x, test_y, _ = crime_data.build_dataset(
    crime_resampled, len_sequence, train_val_shuffle=False
)

# road network data
road_data = dd.RoadData()
road_data.read_road_shp(dir_roads)
road_data.connect_line()
road_data.get_pop_on_roads(dir_local_population, dir_thi_polygon)
road_data.convert_roads_2_network()

# # hyper-param tuning (close after development)
# crime_pred_tuner = mo.CrimePredTuner(model_name='GConvGRU', model_save=dir_cache)
# crime_pred_tuner.build_adj_from_polygons(crime_data.polygon.to_crs('EPSG:4326'), 'Queen')
# crime_pred_tuner.build_dataset(train_x, train_y, val_x, val_y, test_x, test_y)
# crime_pred_tuner.run_study()

# # train model recursively (close during tests, models have been cached)
# crime_pred = mo.CrimePred()
# crime_pred.build_adj_from_polygons(crime_data.polygon.to_crs('EPSG:4326'), 'Queen')
# for i in range(test_x.shape[0]):
#     train_x, train_y, val_x, val_y, test_x, test_y = dd.shift_samples_test_2_train(
#         train_x, train_y, val_x, val_y, test_x, test_y, i, train_val_shuffle=False
#     )
#     if i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 19, 20, 23, 24, 25, 26, 27, 28]:
#         continue
#     for _ in range(5):
#         crime_pred.build_dataset(train_x, train_y, val_x, val_y, test_x, test_y)
#         crime_pred.build_model(1, 1, model_name='GConvGRU', K=2)
#         crime_pred.train_model(8, 0.005, 100, dir_cache, test_loc=i)

# record recursively
metrics_ml = []
metrics_naive = []
crime_pred = mo.CrimePred()
crime_pred.build_adj_from_polygons(crime_data.polygon.to_crs('EPSG:4326'), 'Queen')
for i in range(test_x.shape[0]):
    train_x, train_y, val_x, val_y, test_x, test_y = dd.shift_samples_test_2_train(
        train_x, train_y, val_x, val_y, test_x, test_y, i,
    )
    crime_pred.build_dataset(train_x, train_y, val_x, val_y, test_x, test_y)
    pred, true = crime_pred.pred_crime_test_set(dir_cache, i)
    pred = pred[0, :, :]
    true = true[0, :, :]
    metrics_ml.append(ev.cal_metrics(pred, true, 'ml'))
    metrics_naive.append(ev.cal_metrics(ev.build_baseline_ave([train_x, val_x]), true, 'naive'))

    # # cam placement
    # sim = mo.SensorPlacement(
    #     crime_pred.convert_pred_2_int(pred, crime_pred.test_dataset.mean_max_x),
    #     road_data.road_lines, road_data.road_network, sim_iteration_num, sim_random_seed
    # )
    # sim.run_sim()
    # sim.place_multiple_cameras(cam_count=sim_cam_num)  # pre_located_cam=sim_pre_located_cam

ev.aggr_metrics(metrics_ml, 'ml')
ev.aggr_metrics(metrics_naive, 'naive')



print()

