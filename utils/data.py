import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime


class CrimeData:
    def __init__(self,):
        self.time_interval = None
        self.crime = None
        self.polygon = None
        self.test_index = None

    def import_thi_polygon(self, polygon_dir, crs):
        polygon = gpd.read_file(polygon_dir).set_crs(crs)
        polygon = polygon.rename(columns={'OBJECTID': 'id',})
        polygon = polygon[['id', 'geometry']]
        self.polygon = polygon

    def import_crime(self, data_dir, crs):
        df = pd.read_csv(data_dir)
        df = df.rename(columns={'TimeOccur': 'time', 'GeoX': 'lon', 'GeoY': 'lat', 'Type': 'type'})
        df['id'] = range(1, len(df) + 1)
        df['time'] = df['time'].apply(lambda x: datetime.strptime(str(x), '%m/%d/%Y %H:%M:%S'))
        df = df.set_index('time')
        df['crime_number'] = [0] * len(df)
        df['geometry'] = df.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
        gdf = gpd.GeoDataFrame(df, geometry='geometry')
        gdf = gdf.set_crs(crs)
        self.crime = gdf
        return gdf

    def resample_crime(self, time_interval, delete_no_crime_cell=False):
        self.time_interval = time_interval
        crime = self.crime.copy()
        crime_groups = crime.resample(time_interval, label='right', closed='right', origin='end')
        if not delete_no_crime_cell:
            crime_resampled = self.resample_crime_space(crime_groups)
        else:
            crime_resampled = self.resample_crime_space(crime_groups)
            zero_crime_index = self.get_empty_polygon(crime_resampled)
            print(f'{zero_crime_index[0].size / crime_resampled.shape[1] * 100:.2f}% of no-crime polygons were removed.')
            crime_resampled = np.delete(crime_resampled, zero_crime_index, axis=1)
        crime_resampled = crime_resampled[1:]  # remove the first which might have insufficient data
        print(f'Shape of each crime map is {crime_resampled.shape[1:]}. The first crime map removed.')
        return crime_resampled

    def resample_crime_space(self, crime_groups):
        group_list = []
        for t, gdf_point in crime_groups:
            assert gdf_point.crs == self.polygon.crs
            gdf_join = self.polygon.sjoin(gdf_point, how='left', predicate='contains')
            crime_in_polygons = gdf_join.groupby('id_left').count()['crime_number'].values
            group_list.append(crime_in_polygons)
        crime_resampled = np.array(group_list)
        return crime_resampled

    def get_empty_polygon(self, crime_resampled):
        crime_in_polygons = np.sum(crime_resampled, axis=0)
        zero_index = np.where(crime_in_polygons == 0)
        return zero_index

    def build_dataset(
            self, crime_sampled, sequence_length,
            random_index=False, train_split=0.8, val_split=0.1, train_val_shuffle=False
    ):
        dataset = []
        sequence = []
        for c_map in crime_sampled:
            sequence.append(c_map)  # store raster
            if len(sequence) == sequence_length:  # if the sequence is full
                dataset.append(sequence)  # add sequence to dataset
                sequence = sequence[1:]  # delete the first crime map
        dataset = np.asarray(dataset)
        dataset_x = dataset[:-1]
        dataset_y = crime_sampled[sequence_length:]

        indexes = np.arange(dataset_x.shape[0])
        if random_index:
            np.random.shuffle(indexes)
        train_index = indexes[: int(train_split * dataset_x.shape[0])]
        val_index = indexes[int(train_split * dataset_x.shape[0]): int((train_split + val_split) * dataset_x.shape[0])]
        test_index = indexes[int((train_split + val_split) * dataset_x.shape[0]):]
        self.test_index = test_index

        if train_val_shuffle:
            train_val_index = np.concatenate((train_index, val_index))
            np.random.shuffle(train_val_index)
            train_index = train_val_index[:train_index.shape[0]]
            val_index = train_val_index[train_index.shape[0]:]

        train_x = dataset_x[train_index][:, :, :, np.newaxis]
        train_y = dataset_y[train_index][:, np.newaxis, :, np.newaxis]
        val_x = dataset_x[val_index][:, :, :, np.newaxis]
        val_y = dataset_y[val_index][:, np.newaxis, :, np.newaxis]
        test_x = dataset_x[test_index][:, :, :, np.newaxis]
        test_y = dataset_y[test_index][:, np.newaxis, :, np.newaxis]
        x_4_pred = dataset[-1:][:, :, :, np.newaxis]

        print(f'Shape of train_x is {train_x.shape}')
        print(f'Shape of train_y is {train_y.shape}')
        print(f'Shape of val_x is {val_x.shape}')
        print(f'Shape of val_x is {val_y.shape}')
        print(f'Shape of test_x is {test_x.shape}')
        print(f'Shape of test_y is {test_y.shape}')
        print(f'Shape of x_4_pred is {x_4_pred.shape}')
        return train_x, train_y, val_x, val_y, test_x, test_y, x_4_pred


def read_camera(addr, crs=None):
    df = pd.read_csv(addr)
    df = df[df['Technology'].isin([
        'Automated License Plate Readers', 'Camera Registry',
        'Face Recognition', 'Video Analytics',
        'Gunshot Detection', 'Cell-site Simulator',
    ])]
    df = df[['County', 'State', 'lat', 'lon']]
    df = df.rename(columns={'County': 'county', 'State': 'state',})
    geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry)
    if crs is not None:
        gdf = gdf.set_crs(crs)
    return gdf


def geocode_camera(addr, save_dir,):
    import os
    if os.path.exists(save_dir):
        print("Geocoded camera file exists.")
        return
    else:
        df = pd.read_csv(addr)
        df['addr'] = df['City'] + ', ' + df['County'] + ', ' + df['State']

        from geopy import geocoders
        from geopy.exc import GeocoderTimedOut
        from geopy.extra.rate_limiter import RateLimiter
        locator = geocoders.ArcGIS()
        geolocator = RateLimiter(locator.geocode, min_delay_seconds=0.1)

        def geocode(address):
            try:
                location = geolocator(address)
                if location:
                    return location.latitude, location.longitude
                else:
                    return None, None
            except GeocoderTimedOut:
                return None, None

        df[['lat', 'lon']] = df['addr'].apply(lambda x: pd.Series(geocode(x)))
        df = df.drop(columns=['addr'])
        df.to_csv(save_dir, index=False)
        return df


def read_population(addr):
    df = pd.read_csv(addr)
    df = df.iloc[1:]
    df = df[['GEO_ID', 'B01003_001E',]]
    df = df.rename(columns={'GEO_ID': 'geo_id', 'B01003_001E': 'pop',})
    df['pop'] = df['pop'].astype(int)
    return df


def read_county(dir):
    gdf = gpd.read_file(dir)
    gdf = gdf[gdf['STATEFP'].isin([
        '01', '04', '05', '06', '08', '09', '10', '11', '12', '13',
        '16', '17', '18', '19', '20', '21', '22', '23', '24', '25',
        '26', '27', '28', '29', '30', '31', '32', '33', '34', '35',
        '36', '37', '38', '39', '40', '41', '42', '44', '45', '46',
        '47', '48', '49', '50', '51', '53', '54', '55', '56'
    ])]  # CONUS
    gdf = gdf[['GEOIDFQ', 'geometry']]
    gdf = gdf.rename(columns={'GEOIDFQ': 'geo_id',})
    return gdf


def match_census_county(census, county):
    gdf = county.merge(census, on='geo_id', how='left')
    return gdf


def get_top_counties(gdf):
    gdf = gdf.sort_values(by='pop', ascending=False)
    gdf['cum_pop'] = gdf['pop'].cumsum()
    gdf['cum_per_pop'] = gdf['cum_pop'] / gdf['pop'].sum()
    gdf['if_top'] = [False] * len(gdf)
    gdf.loc[gdf['cum_per_pop'] <= 0.5, 'if_top'] = True
    return gdf


def shift_samples_test_2_train(train_x, train_y, val_x, val_y, test_x, test_y, no_shift, train_val_shuffle=False):
    if no_shift != 0:
        val_x = np.concatenate((val_x, test_x[:1]), axis=0)
        val_y = np.concatenate((val_y, test_y[:1]), axis=0)
        test_x = test_x[1:]
        test_y = test_y[1:]

        train_x = np.concatenate((train_x, val_x[:1]), axis=0)
        train_y = np.concatenate((train_y, val_y[:1]), axis=0)
        val_x = val_x[1:]
        val_y = val_y[1:]
    else:
        pass

    if train_val_shuffle:
        combined_x = np.concatenate((train_x, val_x), axis=0)
        combined_y = np.concatenate((train_y, val_y), axis=0)
        indices = np.arange(combined_x.shape[0])
        np.random.shuffle(indices)
        shuffled_x = combined_x[indices]
        shuffled_y = combined_y[indices]
        train_x = shuffled_x[:len(train_x)]
        train_y = shuffled_y[:len(train_y)]
        val_x = shuffled_x[len(train_x):]
        val_y = shuffled_y[len(train_y):]

    return train_x, train_y, val_x, val_y, test_x, test_y


if __name__ == '__main__':

    pass




