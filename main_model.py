from config import *
import data as dd
import model as mo
import numpy as np

crime_data = dd.CrimeData()
crime_data.import_thi_polygon(dir_thi_polygon, crs_polygon)
crime_data.import_crime(dir_crime, crs_point)
crime_resampled = crime_data.resample_crime(time_interval)
train_x, train_y, val_x, val_y, test_x, test_y, x_4_pred = crime_data.build_dataset(
    crime_resampled, len_sequence,
)

# crime_pred = mo.CrimePred()
# crime_pred.build_adj_from_polygons(crime_data.polygon.to_crs('EPSG:4326'), 'Queen')
# crime_pred.build_dataset(train_x, train_y, val_x, val_y, test_x, test_y)
# crime_pred.build_model(1, 1,)
# crime_pred.train_model(16, 0.002, 500, dir_cache)
# pred, true = crime_pred.pred_crime_test_set(dir_cache)

crime_pred_tuner = mo.CrimePredTuner()
crime_pred_tuner.build_adj_from_polygons(crime_data.polygon.to_crs('EPSG:4326'), 'Queen')
crime_pred_tuner.build_dataset(train_x, train_y, val_x, val_y, test_x, test_y)
crime_pred_tuner.run_study()


print()
