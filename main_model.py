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
    crime_resampled, len_sequence,
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
    for _ in range(10):
        train_x, train_y, val_x, val_y, test_x, test_y = dd.shift_samples_test_2_train(
            train_x, train_y, val_x, val_y, test_x, test_y, i
        )
        crime_pred.build_dataset(train_x, train_y, val_x, val_y, test_x, test_y)
        crime_pred.build_model(1, 1, model_name='GConvGRU', K=2)
        crime_pred.train_model(8, 0.003, 1, dir_cache, test_loc=i)

# test in one batch
crime_pred = mo.CrimePred()
crime_pred.build_adj_from_polygons(crime_data.polygon.to_crs('EPSG:4326'), 'Queen')
crime_pred.build_dataset(train_x, train_y, val_x, val_y, test_x, test_y)
pred, true = crime_pred.pred_crime_test_set(dir_cache)
ev.cal_metrics(pred, true, 'ml')
ev.cal_metrics(ev.build_baseline_ave([train_x, val_x]), true, 'naive')
print()
