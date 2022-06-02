import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime
import matplotlib.pyplot as plt
from shapely.geometry import Point

def Data_import(df, addr_c = './data/DataForDemo.csv', ti = 'TimeOccur', gx = 'GeoX', gy = 'GeoY', ty = 'Type', \
                summary = False, SelectType = None):
    # USE: parse data
    
    #df = pd.read_excel(addr_c, names=[ti, gx, gy, ty])
    TimeOccurTemp = df['TimeOccur']
    TimeOccur = []
    format = '%m/%d/%Y %H:%M:%S'
    for time in TimeOccurTemp:
        t = datetime.strptime(str(time), format)
        TimeOccur.append(t)
    df.loc[:,(ti)] = TimeOccur
    dff = df.drop(columns=[ti,gx,gy])
    df = df.set_index(ti)

    df.insert(df.shape[1], 'CrimeNumber', list(np.zeros((df.shape[0],), dtype=int))) # Add crime amount column
    #df = df.iloc[::-1] # Reverse the timeline
    
    if summary == True:
        print(dff.apply(pd.value_counts))
        dff.apply(pd.value_counts).plot(kind='bar')
        plt.show()

    if SelectType == None:
        return df
    else:
        dfSelected = df[df[ty].isin(SelectType)]
        dfOthers = df[~df[ty].isin(SelectType)]
        return df, dfSelected, dfOthers

def CrimeRaster(df, addr_g = './data/ThiessenPolygons.shp', crs_g = 'EPSG:2240', gx = 'GeoX', gy = 'GeoY'):
    pf = gpd.read_file(addr_g).set_crs(crs_g)
    CrimeRaster = np.zeros([1, pf.shape[0]])   
    Xlist = df[gx].tolist()
    Ylist = df[gy].tolist()
    ccc = 0
    for X, Y in zip(Xlist, Ylist):            
        point = Point(X, Y)
        series = pf.contains(point)
        try: 
            loc = series[series == True].index.tolist()[0]
        except:
            #print("NOTE: There are crime points out of polygon boundary.")
            continue      
        CrimeRaster[0, loc] += 1 
    return CrimeRaster

def FindEmpty (df, size_resample):
    # Find the location of cells without any crime in past years
    dfSampled = df.resample(size_resample).apply(CrimeRaster)
    print('\nThe shape of each crime map is ', dfSampled.tolist()[0].shape, '. \n') 
    CrimeOverlap = sum(dfSampled.tolist())
    zeroloc = np.where(CrimeOverlap == 0) 
    return zeroloc, CrimeOverlap
            
def Resampling(df, zeroloc, CrimeOverlap, size_resample, Control_DeleteNoCrimeCell = True):
    # Cut points into periods and cells
    # Detect cells with no crime and REDO resampling if needed
    if Control_DeleteNoCrimeCell == False:
        dfSampled = df.resample(size_resample).apply(CrimeRaster)
        print('\nThe shape of each crime map is ', dfSampled.tolist()[0].shape, '. \n') 
        
    else:
        def CrimeRaster_clean(df, zeroloc = zeroloc):
            data = CrimeRaster(df)
            data = np.delete(data, zeroloc[1])
            return data
        dfSampled = df.resample(size_resample).apply(CrimeRaster_clean)
        
        NoCrimeRatio = zeroloc[1].size / CrimeOverlap.size
        print(NoCrimeRatio * 100, '% of the cells have no crime record. \n')
        print('The shape of the cleaned crime map is', dfSampled.tolist()[0].shape, '. \n')  
    return dfSampled

def Data_resampler(df, size_resample = '7d', SelectType = None, dfSelected = None, dfOthers = None):
    if SelectType == None:
        zeroloc, CrimeOverlap = FindEmpty (df, size_resample)
        dfSampled = Resampling(df, zeroloc, CrimeOverlap, size_resample)
        return dfSampled
    else:
        zeroloc, CrimeOverlap = FindEmpty (df, size_resample)
        dfSelectedSampled = Resampling(dfSelected, zeroloc, CrimeOverlap, size_resample)
        dfOthersSampled = Resampling(dfOthers, zeroloc, CrimeOverlap, size_resample)
        # Align the length of two df
        if dfSelectedSampled.index.shape[0] == dfOthersSampled.index.shape[0]:
            pass
        elif dfSelectedSampled.index.shape[0] < dfOthersSampled.index.shape[0]:
            dfOthersSampled = dfOthersSampled.loc[dfSelectedSampled.index]
        else:
            dfSelectedSampled = dfSelectedSampled.loc[dfOthersSampled.index]
        return dfSelectedSampled, dfOthersSampled

def Data_loader(dfSampled, len_sequence, per_train = 0.9, per_val = 0.075, production_env = True):
    Data = []
    Sequence = []

    for i in dfSampled.values:
        Sequence.append(i)  # Store raster
        if len(Sequence) == len_sequence: # If this sequence is fully filled
            Data.append(Sequence) # Add sequence to dataset
            Sequence = Sequence[1: ] # Delete the first crime map

    for data, t in zip(Data, np.arange(0,len(Data),1)):
        Data[t] = list(data)
    Data = np.array(Data)
    
    DataX = Data[ :-1]
    DataY = dfSampled.tolist()[len_sequence: ]
    DataY = np.asarray(DataY)

    # Training, validation, and test data
    indexes = np.arange(DataX.shape[0])[::-1]
    #np.random.shuffle(indexes)
    train_index = indexes[: int(per_train * DataX.shape[0])]
    val_index = indexes[int(per_train * DataX.shape[0]): int((per_train + per_val) * DataX.shape[0])]
    test_index = indexes[int((per_train + per_val) * DataX.shape[0]) :]

    TrainX = DataX[train_index]
    TrainY = DataY[train_index]
    ValX = DataX[val_index]
    ValY = DataY[val_index]
    TestX = DataX[test_index]
    TestY = DataY[test_index]
    
    if production_env == True:
    # If this system is in real use, set TestX as the most recent week and set TestY as NaN
    # In this case, evaluation cannot run.
        TestX = Data[0]
        TestX = TestX[np.newaxis, :, :]

        TestY = np.empty((1, TestX.shape[-1]))
        TestY[:] = np.nan

    TrainX = TrainX[:,:,np.newaxis,:, np.newaxis]
    TrainY = TrainY[:,np.newaxis,:, np.newaxis]
    ValX = ValX[:,:,np.newaxis,:, np.newaxis]
    ValY = ValY[:,np.newaxis,:, np.newaxis]
    TestX = TestX[:,:,np.newaxis,:, np.newaxis]
    TestY = TestY[:,np.newaxis,:, np.newaxis]

    return TrainX, TrainY, ValX, ValY, TestX, TestY

def Load_data(dfSampled = None, len_sequence = 16, SelectType = None, dfSelectedSampled = None, dfOthersSampled = None):
    if SelectType == None:
        TrainX, TrainY, ValX, ValY, TestX, TestY = Data_loader(dfSampled, len_sequence)
        print('\nThe shape of TrianX is', TrainX.shape, '. \n')
        print('The shape of TrainY is', TrainY.shape, '. \n')
        print('The shape of ValX is', ValX.shape, '. \n')
        print('The shape of ValY is', ValY.shape, '. \n')
        print('The shape of TestX is', TestX.shape, '. \n')
        print('The shape of TestY is', TestY.shape, '. \n')
        return TrainX, TrainY, ValX, ValY, TestX, TestY
    else:
        TrainX_S, TrainY_S, ValX_S, ValY_S, TestX_S, TestY_S = Data_loader(dfSelectedSampled, len_sequence)
        TrainX_O, TrainY_O, ValX_O, ValY_O, TestX_O, TestY_O = Data_loader(dfOthersSampled, len_sequence)
        TrainX = np.concatenate((TrainX_S, TrainX_O), axis=-1)
        TrainY = TrainY_S
        ValX = np.concatenate((ValX_S, ValX_O), axis=-1)
        ValY = ValY_S
        TestX = np.concatenate((TestX_S, TestX_O), axis=-1)
        TestY = TestY_S
        print('\nThe shape of TrianX is', TrainX.shape, '. \n')
        print('The shape of TrainY is', TrainY.shape, '. \n')
        print('The shape of ValX is', ValX.shape, '. \n')
        print('The shape of ValY is', ValY.shape, '. \n')
        print('The shape of TestX is', TestX.shape, '. \n')
        print('The shape of TestY is', TestY.shape, '. \n')
        return TrainX, TrainY, ValX, ValY, TestX, TestY



if __name__ == '__main__':
    # Import data, including crime records and polygon shapefile    
    df = Data_import()
    # Do resampling
    dfSampled = Data_resampler(df)
    # Build sequences and datasets
    TrainX, TrainY, ValX, ValY, TestX, TestY = Load_data(dfSampled)