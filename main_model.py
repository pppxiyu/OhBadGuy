from config import *
from utils import data as dd
from utils import evaluation as ev
import model as mo

# datasets
crime_data = dd.CrimeData()
crime_data.import_thi_polygon(dir_thi_polygon, crs_polygon)
crime_data.import_crime(dir_crime, crs_point)
crime_resampled = crime_data.resample_crime(time_interval)
train_x, train_y, val_x, val_y, test_x, test_y, _ = crime_data.build_dataset(
    crime_resampled, len_sequence, train_val_shuffle=False
)

# # hyper-param tuning
# crime_pred_tuner = mo.CrimePredTuner(model_name='GConvGRU', model_save=dir_cache)
# crime_pred_tuner.build_adj_from_polygons(crime_data.polygon.to_crs('EPSG:4326'), 'Queen')
# crime_pred_tuner.build_dataset(train_x, train_y, val_x, val_y, test_x, test_y)
# crime_pred_tuner.run_study()

# train model recursively
crime_pred = mo.CrimePred()
crime_pred.build_adj_from_polygons(crime_data.polygon.to_crs('EPSG:4326'), 'Queen')
for i in range(test_x.shape[0]):
    if i in [0, 1]:
        pass
    train_x, train_y, val_x, val_y, test_x, test_y = dd.shift_samples_test_2_train(
        train_x, train_y, val_x, val_y, test_x, test_y, i, train_val_shuffle=False
    )
    for _ in range(10):
        crime_pred.build_dataset(train_x, train_y, val_x, val_y, test_x, test_y)
        crime_pred.build_model(1, 1, model_name='GConvGRU', K=2)
        crime_pred.train_model(8, 0.003, 100, dir_cache, test_loc=i)
        pass

# test recursively
metrics_ml = []
metrics_naive = []
crime_pred = mo.CrimePred()
crime_pred.build_adj_from_polygons(crime_data.polygon.to_crs('EPSG:4326'), 'Queen')
for i in range(test_x.shape[0]):
    train_x, train_y, val_x, val_y, test_x, test_y = dd.shift_samples_test_2_train(
        train_x, train_y, val_x, val_y, test_x, test_y, i
    )
    crime_pred.build_dataset(train_x, train_y, val_x, val_y, test_x, test_y)
    pred, true = crime_pred.pred_crime_test_set(dir_cache, i)
    pred = pred[0, :, :]
    true = true[0, :, :]
    metrics_ml.append(ev.cal_metrics(pred, true, 'ml'))
    metrics_naive.append(ev.cal_metrics(ev.build_baseline_ave([train_x, val_x]), true, 'naive'))
ev.aggr_metrics(metrics_ml, 'ml')
ev.aggr_metrics(metrics_naive, 'naive')


print()
