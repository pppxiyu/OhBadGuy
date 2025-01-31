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

crime_pred = mo.CrimePred()
crime_pred.build_adj_from_polygons(crime_data.polygon.to_crs('EPSG:4326'), 'Queen')
crime_pred.build_dataset(train_x, train_y, val_x, val_y, test_x, test_y)
crime_pred.build_model(1, 1,)
crime_pred.train_model(16, 0.002, 1, dir_cache)
pred, true = crime_pred.pred_crime_test_set(dir_cache)



# predict
# prediction = cp.predict(testX, testY, 0, lap = lapacian)
# prediction = np.rint(prediction).tolist()
# cp.polygonUpdate(prediction)

# Visualization
# # Point map
# cp.crimePointMap(np.random.choice(range(testX.shape[0]), size=1)[0])
# # Polygon map
# cp.crimePolygonMap()
# Point and polygon map
# fig = cp.crimePolygonPointMap()
# # get static
# cp.polygonUpdate(cp.getRule0Plan(7).tolist())
# cp.crimePolygonMap()

# Actually predict (only X no Y) with visualization
# the second latestX is just for fill the position, the testY does not exist
prediction = crime_pred.predict(X4Predict, np.empty(X4Predict.shape), 0, lap = lapacian)
crime_pred.polygonUpdate(prediction)
# cp.crimePolygonMap()

# y = np.concatenate((trainY,valY), axis=0)
# y_mean = np.mean(y, axis=0).tolist()
# prediction = y_mean
# cp.polygonUpdate(prediction)


print()
