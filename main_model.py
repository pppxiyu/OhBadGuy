from config import *
from utils import data as dd
from utils import eval as ev
from utils import vis
import model as mo
import os

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

# # hyper-param tuning (close after fine-tuning)
# crime_pred_tuner = mo.CrimePredTuner(model_name=model_name, model_save=dir_cache)
# crime_pred_tuner.build_adj_from_polygons(crime_data.polygon.to_crs('EPSG:4326'), 'graph', road_geo=road_data.road_lines)
# crime_pred_tuner.build_dataset(train_x, train_y, val_x, val_y, test_x, test_y)
# crime_pred_tuner.run_study()

# # train model recursively (close during tests, models have been cached)
# crime_pred = mo.CrimePred()
# crime_pred.build_adj_from_polygons(crime_data.polygon.to_crs('EPSG:4326'), 'graph', road_geo=road_data.road_lines)
# test_x_iterate = test_x.copy()
# test_y_iterate = test_y.copy()
# for i in range(test_x.shape[0] - 1):
#     i += 1
#     print(train_x.shape[0], val_x.shape[0], test_x_iterate.shape[0])
#     train_x, train_y, val_x, val_y, test_x_iterate, test_y_iterate = dd.shift_samples_test_2_train(
#         train_x, train_y, val_x, val_y, test_x_iterate, test_y_iterate, i, train_val_shuffle=False
#     )
#     for _ in range(20):
#         crime_pred.build_dataset(train_x, train_y, val_x, val_y, test_x_iterate, test_y_iterate)

#         # crime_pred.build_model(1, 1, model_name=model_name, K=gnn_k)
#         # crime_pred.train_model(8, 0.005, 100, dir_cache, test_loc=i)  # full model
#         # # crime_pred.train_model(8, 0.003, 100, dir_cache, test_loc=i)  # full model

#         crime_pred.build_model(1, 1, model_name=model_name, K=gnn_k)
#         crime_pred.train_model(8, 0.001, 100, dir_cache, test_loc=i)  # noConv model

#         # crime_pred.build_model(1, 1, model_name=model_name, K=1)
#         # crime_pred.train_model(8, 0.013, 100, dir_cache, test_loc=i)  # noRes model

# record recursively
metrics_ml = []
metrics_naive = []
crime_pred = mo.CrimePred()
crime_pred.build_adj_from_polygons(crime_data.polygon.to_crs('EPSG:4326'), 'graph', road_geo=road_data.road_lines)
test_x_iterate = test_x.copy()
test_y_iterate = test_y.copy()

# # vis baseline static density map
# vis.density_crime_map(
#     ev.build_baseline_ave([train_x, val_x]).flatten(), crime_pred.adj_matrix,
#     dir_city_boundary, 
#     roads=road_data.road_lines.copy(),
#     polygons=crime_data.polygon.copy(),
#     spatial_resolution=100, n_layers=40, save_path=f'{dir_figs}/density_map_static.png'
# )

for i in range(test_x.shape[0] - 1):
    i += 1
    train_x, train_y, val_x, val_y, test_x_iterate, test_y_iterate = dd.shift_samples_test_2_train(
        train_x, train_y, val_x, val_y, test_x_iterate, test_y_iterate, i,
    )
    crime_pred.build_dataset(train_x, train_y, val_x, val_y, test_x_iterate, test_y_iterate)
    pred, true = crime_pred.pred_crime_test_set(dir_cache, i)
    metrics_ml.append(ev.cal_metrics(pred[0, :, :], true[0, :, :], 'ml', adj_matrix=crime_pred.adj_matrix))
    metrics_naive.append(ev.cal_metrics(ev.build_baseline_ave([train_x, val_x]), true[0, :, :], 'naive', adj_matrix=crime_pred.adj_matrix))

    # if i in [1, 8, 14]:
    #     # vis: dynamics vs static density map /pred map 
    #     folder_path = f"{dir_figs}/density_map_diff"
    #     os.makedirs(folder_path, exist_ok=True)
    #     vis.density_crime_map(
    #         [ev.build_baseline_ave([train_x, val_x]).flatten(), pred[0, :, :].flatten()], 
    #         crime_pred.adj_matrix,
    #         dir_city_boundary, 
    #         roads=road_data.road_lines.copy(),
    #         polygons=crime_data.polygon.copy(),
    #         spatial_resolution=100, n_layers=40, save_path=f'{folder_path}/density_map_diff_dynamic_{i}.png'
    #     )
    #     vis.density_crime_map(
    #         [ev.build_baseline_ave([train_x, val_x]).flatten(), true[0, :, :].flatten()], 
    #         crime_pred.adj_matrix,
    #         dir_city_boundary, 
    #         roads=road_data.road_lines.copy(),
    #         polygons=crime_data.polygon.copy(),
    #         spatial_resolution=100, n_layers=40, save_path=f'{folder_path}/density_map_diff_static_{i}.png'
    #     )

    #     # vis: crime density
    #     folder_path = f"{dir_figs}/density_map_pred"
    #     os.makedirs(folder_path, exist_ok=True)
    #     vis.density_crime_map(
    #         pred[0, :, :].flatten(), 
    #         crime_pred.adj_matrix,
    #         dir_city_boundary, 
    #         roads=road_data.road_lines.copy(),
    #         polygons=crime_data.polygon.copy(),
    #         spatial_resolution=100, 
    #         save_path=f'{folder_path}/density_map_pred_{i}.png',
    #         n_layers=10, power_transform=2,  # increase power and reduce layers to amplify the nuances 
    #     )

    #     # vis: crime density sequence
    #     vis.density_crime_map_sequence(
    #         pred[0, :, :].flatten(), 
    #         dir_city_boundary, 
    #         roads=road_data.road_lines.copy(),
    #         polygons=crime_data.polygon.copy(),
    #         save_path=f'{folder_path}/density_map_pred_sequence_{i}.png',
    #     )

    # if i in [1, 8, ]:
    #     pop_values = dict(road_data.road_network.nodes(data='pop'))
    #     p_start = {k: v/sum(pop_values.values()) for k, v in pop_values.items()}

    #     probabilities = (pred[0, :, 0] - pred[0, :, 0].min()) / (pred[0, :, 0].max() - pred[0, :, 0].min())
    #     probabilities = probabilities / probabilities.sum()
    #     crime_values = {i + 1: probabilities[i] for i in range(len(probabilities))}
    #     p_end = {k: v/sum(crime_values.values()) for k, v in crime_values.items()}

    #     optimizer = mo.SensorPlacement(road_data.road_network, p_start, p_end,)
    #     selected_sensors, selected_sensors_iterations = optimizer.place_sensors(
    #         num_sensors=10 , verbose=True, get_iterations=True
    #     )
    #     if i in [8, ]:
    #         # # vis: marginal gain of centrality
    #         vis.line_marginal_gain_centrality(
    #             optimizer.get_iteration_scores()
    #             , save_path=f'{dir_figs}/line_marginal_gain_centrality_week_{i}.png'
    #         )
    #     if i in [1, ]:
    #         # # vis: opt process of sensor placement
    #         for ii, plc_iter in enumerate(selected_sensors_iterations):
    #             vis.map_placement(
    #                 selected_sensors[:(ii + 1)], 
    #                 pred[0, :, :].flatten(), 
    #                 dir_city_boundary, 
    #                 roads=road_data.road_lines.copy(),
    #                 polygons=crime_data.polygon.copy(),
    #                 placement_candidate=list(plc_iter.keys())[1:],
    #                 save_path=f'{dir_figs}/opt_process/plc_iteration_{ii}.png',
    #             )

    if i in [1, 8, 14]:
        pop_values = dict(road_data.road_network.nodes(data='pop'))
        p_start = {k: v/sum(pop_values.values()) for k, v in pop_values.items()}

        probabilities = (pred[0, :, 0] - pred[0, :, 0].min()) / (pred[0, :, 0].max() - pred[0, :, 0].min())
        probabilities = probabilities / probabilities.sum()
        crime_values = {i + 1: probabilities[i] for i in range(len(probabilities))}
        p_end = {k: v/sum(crime_values.values()) for k, v in crime_values.items()}

        optimizer = mo.SensorPlacement(road_data.road_network.copy(), road_data.road_lines.copy(), p_start, p_end,)

        # # # vis: sensor placements over weeks (no direction)
        # selected_sensors, selected_sensors_iterations = optimizer.place_sensors(
        #     20 , verbose=True, get_iterations=True)
        # vis.map_placement(
        #     selected_sensors, 
        #     pred[0, :, :].flatten(), 
        #     dir_city_boundary, 
        #     roads=road_data.road_lines.copy(),
        #     polygons=crime_data.polygon.copy(),
        #     draw_polygon=True,
        #     save_path=f'{dir_figs}/plc_over_weeks/plc_week_{i}.png',)

        # # # vis: sensor placements over weeks (with direction)
        # selected_sensors, selected_sensors_iterations = optimizer.place_sensors_w_directions(
        #     20 , verbose=True, get_iterations=True)
        # vis.map_placement_directional(
        #     selected_sensors, optimizer.node_directions,
        #     pred[0, :, :].flatten(), 
        #     dir_city_boundary, 
        #     roads=road_data.road_lines.copy(),
        #     polygons=crime_data.polygon.copy(),
        #     draw_polygon=True,
        #     save_path=f'{dir_figs}/plc_over_weeks/plc_directional_week_{i}.png',)

        # # vis: sensor placements over weeks (with direction and with existing sensors)
        selected_sensors, selected_sensors_iterations = optimizer.place_sensors_w_directions(
            10 , verbose=True, get_iterations=True, preset_sensors=set([d['Road'] for d in pre_located_cam]) 
            )
        vis.map_placement_directional(
            selected_sensors, optimizer.node_directions,
            pred[0, :, :].flatten(), 
            dir_city_boundary, 
            roads=road_data.road_lines.copy(),
            polygons=crime_data.polygon.copy(),
            draw_polygon=True,
            placement_candidate=set([d['Road'] for d in pre_located_cam]),
            save_path=f'{dir_figs}/plc_over_weeks/plc_directional_w_preset_week_{i}.png',)

# vis: metrics and metrics scatter
ev.aggr_metrics(metrics_ml, 'ml')
ev.aggr_metrics(metrics_naive, 'naive')
vis.scatter_crime_pred_metrics(
    [d['Adapted_Pearson'] for d in metrics_ml], [d['Adapted_Pearson'] for d in metrics_naive], 
    y_label='Weighted Pearson Coefficient', annotate=False, stats_test='mannwhitneyu', alternative='greater',
    save_path=f'{dir_figs}/scatter_crime_pred_pearson.png'
)  

print()

