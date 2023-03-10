import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime
import matplotlib.pyplot as plt
from shapely.geometry import Point

import numpy as np
import math
import plotly.express as px
import plotly.graph_objects as go
from shapely.geometry import Point
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from spektral.layers import GCNConv
from spektral.utils import gcn_filter
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

import networkx as nx
import shapely.ops as ops
import random

class dataProcess:
    def __init__(self, polygonAddress = None, crs = None):
        if polygonAddress != None:
            self.polygon = gpd.read_file(polygonAddress).set_crs(crs)
        self.crimeMap = None
        
    def dataImport(self, address, summary = False):
        # Parse the crime record .csv file from a path
        # Input:
        #    summary (boolen): show a descriptive summary of the data
        #    selectType: a list of selected crime types
        # Output:
        #    parse and organized dataframe  
        df = pd.read_csv(address)
        TimeOccurTemp = df['TimeOccur']
        TimeOccur = []
        format = '%m/%d/%Y %H:%M:%S'
        for time in TimeOccurTemp:
            t = datetime.strptime(str(time), format)
            TimeOccur.append(t)
        df.loc[:,('TimeOccur')] = TimeOccur
        dff = df.drop(columns=['TimeOccur','GeoX','GeoY'])
        df = df.set_index('TimeOccur')
        df.insert(df.shape[1], 'CrimeNumber', list(np.zeros((df.shape[0],), dtype=int))) # Add crime amount column

        if summary == True:
            print(dff.apply(pd.value_counts))
            dff.apply(pd.value_counts).plot(kind='bar')
            plt.show()

        return df

    def crimeRaster(self, df):
        # Count the number of crimes in each polygon
        crimeRaster = np.zeros([1, self.polygon.shape[0]])   
        Xlist = df['GeoX'].tolist()
        Ylist = df['GeoY'].tolist()
        for X, Y in zip(Xlist, Ylist):            
            point = Point(X, Y)
            series = self.polygon.contains(point)
            try: 
                loc = series[series == True].index.tolist()[0]
            except:
                #print("WARNING: There are crime points out of polygon boundary.")
                continue      
            crimeRaster[0, loc] += 1 
        return crimeRaster

    def _findEmpty (self, df):
        # Find the location of cells without any crime in past years
        crimeMap = self.crimeRaster(df)
        self.crimeMap = crimeMap
        print('\nThe shape of each crime map is ', crimeMap.shape, '. \n')  
        zeroloc = np.where(crimeMap == 0) 
        return zeroloc

    def dataResampler(self, df, size_resample, deleteNoCrimeCell = False):
        # Cut points into periods and cells
        # Detect cells with no crime and REDO resampling if needed
        
        zeroloc = self._findEmpty(df)
        self.size_resample = size_resample
        if deleteNoCrimeCell == False:
            def CrimeRaster(df):
                data = self.crimeRaster(df)
                data = np.squeeze(data)
                return data  
            dfSampled = df.resample(size_resample, label='right', closed = 'right', origin = 'end').apply(CrimeRaster)
            dfSampled = dfSampled[1:] # remove the first one which might have insufficient data
            print('\nThe shape of each crime map is ', dfSampled.tolist()[0].shape, '. \n') 

        else:
            def CrimeRaster_clean(df, zeroloc = zeroloc):
                data = self.crimeRaster(df)
                data = np.delete(data, zeroloc[1])
                return data
            dfSampled = df.resample(size_resample, label='right', closed = 'right', origin = 'end').apply(CrimeRaster_clean)
            dfSampled = dfSampled[1:] # remove the first one which might have insufficient data
            
            NoCrimeRatio = zeroloc[1].size / self.crimeMap.size
            print(NoCrimeRatio * 100, '% of the cells have no crime record. \n')
            print('The shape of the cleaned crime map is', dfSampled.tolist()[0].shape, '. \n')   
            
        return dfSampled

    def buildDatasets(self, dfSampled, lenSequence, randomIndex = False):
        data = []
        sequence = []

        for i in dfSampled.values:
            sequence.append(i)  # Store raster
            if len(sequence) == lenSequence: # If this sequence is fully filled
                data.append(sequence) # Add sequence to dataset
                sequence = sequence[1: ] # Delete the first crime map

        data = np.asarray(data)
        dataX = data[ :-1]
        
        dataY = dfSampled.tolist()[lenSequence: ]
        dataY = np.asarray(dataY)

        # Training, validation, and test data
        indexes = np.arange(dataX.shape[0])
        if randomIndex == True:
            np.random.shuffle(indexes)
        trainIndex = indexes[: int(0.8 * dataX.shape[0])]
        valIndex = indexes[int(0.8 * dataX.shape[0]): int(0.9 * dataX.shape[0])]
        global testIndex
        testIndex = indexes[int(0.9 * dataX.shape[0]) :]

        trainX = dataX[trainIndex]
        trainY = dataY[trainIndex]
        valX = dataX[valIndex]
        valY = dataY[valIndex]
        testX = dataX[testIndex]
        testY = dataY[testIndex]
        
        trainX = np.moveaxis(trainX, -1, -2)
        valX = np.moveaxis(valX, -1, -2)
        testX = np.moveaxis(testX, -1, -2)
         
        X4Predict = np.moveaxis(data[-1:], -1, -2) #Used for actual prediction
        
        print('\nThe shape of TrianX is', trainX.shape, '. \n')
        print('The shape of TrainY is', trainY.shape, '. \n')
        print('The shape of ValX is', valX.shape, '. \n')
        print('The shape of ValY is', valY.shape, '. \n')
        print('The shape of TestX is', testX.shape, '. \n')
        print('The shape of TestY is', testY.shape, '. \n')
        print('The shape of X4Predict is', X4Predict.shape, '. \n')
        
        return trainX, trainY, valX, valY, testX, testY, X4Predict


class crimePrediction():
    def __init__(self):
        self.mapboxKey = None
        self.pf = None
        self.crimeMapFull = None
        
    def _rescalePredict(self, predict, crimeMapFull):
        if isinstance(predict, np.ndarray):
            return (predict / np.sum(predict)) * np.sum(crimeMapFull)
        elif tf.is_tensor(predict):
            return (predict / tf.reduce_sum(predict)) * np.sum(crimeMapFull)
        else:
            print('WARNING: Input is not numpy array or tensorflow tensor.')
    
    def _klDivergenceElementWise(self, a, b, y_actual):
        # a and b must be tensorflow tensor
        a = keras.backend.clip(a, keras.backend.epsilon(), 1)
        b = keras.backend.clip(b, keras.backend.epsilon(), 1)
        kl = a * tf.math.log(a / b)
        mask = tf.where(y_actual < 1, tf.ones(tf.shape(y_actual)) * self.lowerMultiplier, y_actual)
        return tf.reduce_sum(tf.math.multiply(mask, kl), axis = -1)

    def _customJSDivergence(self, y_actual, y_pred):
        y_actual = y_actual / tf.reduce_sum(y_actual)
        y_pred = y_pred / tf.reduce_sum(y_pred)
        y_m = (y_actual + y_pred) / 2
        return 0.5 * (self._klDivergenceElementWise(y_actual, y_m, y_actual)) + 0.5 * (self._klDivergenceElementWise(y_pred, y_m, y_actual))
    
    def _soft_round(self, x, alpha, eps = 1e-3):
        # x: `tf.Tensor`. Inputs to the rounding function.
        # alpha: Float or `tf.Tensor`. Controls smoothness of the approximation.
        # eps: Float. Threshold below which `soft_round` will return identity.
        # This guards the gradient of tf.where below against NaNs, while maintaining
        # correctness, as for alpha < eps the result is ignored.
#         alpha_bounded = tf.maximum(alpha, eps)
        alpha_bounded = max(alpha, eps)
        m = tf.floor(x) + .5
        r = x - m
        z = tf.tanh(alpha_bounded / 2.) * 2.
        y = m + tf.tanh(alpha_bounded * r) / z
        # For very low alphas, soft_round behaves like identity
        return tf.where(alpha < eps, x, y, name="soft_round")

    def train(self, TrainX, TrainY, ValX, ValY, lap_train = None, lap_val = None, \
              epochs = 50, batch_size = 6, learning_rate = 0.0001, patienceReduceLR = 3, patienceEarlyStopping = 10, \
              alpha_roundApproximation = 7, lowerMultiplier = 0.8):
        
        self.lowerMultiplier = lowerMultiplier
        
        inputs_sequence = keras.layers.Input(shape = (TrainX.shape[1], TrainX.shape[2]))
        inputs_lapacian = keras.layers.Input(shape = (TrainX.shape[1], TrainX.shape[1]))
        x = GCNConv(TrainX.shape[2], activation='relu', use_bias = True)([inputs_sequence, inputs_lapacian])
        x = keras.layers.Permute((2, 1))(x)
        x = keras.layers.LSTM(128, activation = 'relu')(x)
        x = keras.layers.Dense(128, activation = 'relu')(x)
        x = keras.layers.Dense(64, activation = 'relu')(x)
        x = keras.layers.Dense(32, activation = 'relu')(x)
        x = keras.layers.Dense(TrainX.shape[1], activation = 'relu')(x)
        x = self._rescalePredict(x, self.crimeMapFull)
        outputs = self._soft_round(x, alpha = alpha_roundApproximation)

        model = keras.Model(inputs = [inputs_sequence, inputs_lapacian], outputs = outputs)
        model.compile(optimizer = keras.optimizers.Adam(learning_rate = learning_rate), 
                      loss = self._customJSDivergence,)
        model.summary()
        early_stopping = keras.callbacks.EarlyStopping(monitor = "val_loss", patience = patienceEarlyStopping)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = "val_loss", patience = patienceReduceLR)

        model.fit(
            [TrainX, lap_train],
            TrainY,
            batch_size = batch_size,
            epochs = epochs,
            validation_data = ([ValX, lap_val], ValY),
            callbacks = [early_stopping, reduce_lr],
#             verbose = 0,
        )
        
        return model
         
    def predict(self, testX, testY, dataPointIndex, lap = None):
        self.testWeek = dataPointIndex
        testX_point = testX[dataPointIndex].copy()
        predict = self.trainedModel([np.expand_dims(testX_point, axis = 0), np.expand_dims(lap, axis = 0)])
        predict = np.squeeze(predict)
        return predict.tolist()    
    
    def polygonUpdate(self, predict):          
        predictY_list = []
        for Bi in predict:
            predictY_list.append(str(int(Bi)))
        self.pf['Prediction'] = predictY_list

    def crimePolygonMap(self):
        fig = px.choropleth_mapbox(self.pf, geojson = geo_df_select.geometry,
                                   locations = geo_df_select.index, 
                                   color = "Prediction", 
                                   color_discrete_sequence = px.colors.sequential.Reds,
                                   center = {"lat": 32.606827, "lon": -83.660198},
                                   zoom = 13, opacity = 0.7)
        fig.update_layout(mapbox_style = "dark", 
                          mapbox_accesstoken = self.mapboxKey, 
                          margin = {"r":0,"t":0,"l":0,"b":0}, 
                          showlegend = True, 
                          hovermode = False)
        return fig
        

class simulation(crimePrediction):
    def __init__(self, origins, roadData, G, runNum, randomSeed, polygon = None, mapboxKey = ''):       
        random.seed(randomSeed) #set random seed
        self.runNum = runNum 
        self.origins = origins
        self.roadData = roadData
        self.startPoints = self.buildStartPoints(origins)
        self.G = G
        self.polygon = polygon        
        self.mapboxKey = mapboxKey      
        
    def changeRandomSeedTo(self, randomSeed):
        random.seed(randomSeed)
    
    def buildStartPoints(self, origins):
        start_points = []
        loc = 1
        for item in origins:
            if item != None and item >= 1:
                start_points.append(loc)
            loc += 1 
        return start_points
           
    ##########################################################################
    ############################ Simulation ##################################    
    def _vehicleRoute_fromPredictedLocations(self, node_num):
        #OUTPUT: list of nodes
        randomNode = random.choice(list(self.G.nodes())) # randomly get a node in the graph
        route = nx.shortest_path(self.G, weight = 'weight', source = node_num, target = randomNode)
        return route

    def _vehicleRoute_toPredictedLocations(self, node_num):
        #OUTPUT: list of nodes
        # randomly choose road according to population
        roads = [item[0] for item in list(self.G.nodes(data = 'pop'))]
        pops = [item[1] for item in list(self.G.nodes(data = 'pop'))]
        pops_dist = pops / sum(pops)
        randomNode = random.choices(roads, pops_dist)[0]
        # get route (in a reverse way, from pop dist to predicted crime incident locations)
        route = nx.shortest_path(self.G, weight = 'weight', source = randomNode, target = node_num)
        return route
    
    def _roadsSeperationByDirection(self, connectedRoads, buffer = 0, lost_warning = False):
        # USE: assign the connected roads to the two end points of the road for analysis
        # INPUT: dict, the connected roads for a road
        # OUTPUT: a dict
        
        # get the geometry of this road and the coordinates of this line
        get_line = self.roadData[self.roadData.OBJECTID == list(connectedRoads.keys())[0]].geometry.values[0]
        try:
            coords = get_line.coords
        except: # When lines are disconnected, data problem
            coords = [get_line[0].coords[0], get_line[-1].coords[-1]]
            if lost_warning == True:
                print("Remedial measures are taken to extract end points from road segments. Please note there are problems (i.e., lines are not connected in the same group) in the data, although they are solved for this time.")
        
        # assign the connected roads to the two end points of the road in analysis
        c1 = []
        c2 = []
        for road in list(connectedRoads.values())[0]:
            connectedRoadLine = self.roadData[self.roadData.OBJECTID == road].geometry.values[0]
            if connectedRoadLine.distance(Point(coords[0])) <= buffer: # if the connected raod is closer to the cor 0
                c1.append(road)
            elif connectedRoadLine.distance(Point(coords[-1])) <= buffer: # if the connected raod is closer to the cor 1
                c2.append(road)
            else:
                if lost_warning == True:
                    print('Road', road, 'is not attached to any vetice of the target road', list(connectedRoads.keys())[0], '.')
        return {'v1': c1, 'v2': c2}

    def _placementWeight(self, recordsByRoads, roadClustersByDirection):
        # USE: Calculate the weight of each placement using simulation records and road cluster information
        # INPUT: recordsByRoads - all simulation record grouped by roads
        #        roadClustersByDirection - road cluster information for each road    

        placement_weight = []
        for i in range(len(recordsByRoads)):
            record = recordsByRoads[i + 1] 
            roadset_clustered = roadClustersByDirection[i + 1]
            
            # Calculate average detection number for each road pair 
            # (# of vehicles detected among all iteration / total # of iteration)
            roads_v1 = roadset_clustered['v1']
            roads_v2 = roadset_clustered['v2']
            detection_num_v1_temp = [[i['iteration'], i['origin'], i['sequence on origin']] \
                                     for i in record if (i['step1'] in roads_v1 or i['step2'] in roads_v2)] # start is on v1 or end is on v2
            detection_num_v2_temp = [[i['iteration'], i['origin'], i['sequence on origin']] \
                                     for i in record if (i['step1'] in roads_v2 or i['step2'] in roads_v1)] # start is on v2 or end is on v1
            # multiplying by 2 as records are counted twice for each road (in and out)
            ave_detection_num_v1 = len(detection_num_v1_temp) / (2 * self.runNum) 
            ave_detection_num_v2 = len(detection_num_v2_temp) / (2 * self.runNum) 
            
            # Tidy up the data
            out1 = {'Road': i + 1, 'Inflow roads':roads_v1, 'Outflow roads': roads_v2, 'Number of detected vehicles': ave_detection_num_v1}
            out2 = {'Road': i + 1, 'Inflow roads': roads_v2, 'Outflow roads': roads_v1, 'Number of detected vehicles': ave_detection_num_v2}
                  
            placement_weight.append(out1)
            placement_weight.append(out2)
        placement_weight_sorted = sorted(placement_weight, reverse = True, key = lambda x: x['Number of detected vehicles'])
        return placement_weight_sorted

    def getRoadSeperation(self):
        roadClustersByDirection = {k + 1: None for k in range(self.G.number_of_nodes())}
        for road in range(self.G.number_of_nodes()):
            roadsNeighbor = list(nx.neighbors(self.G, road + 1))
            roadsConnected = {road + 1: roadsNeighbor}
            roadset_clustered = self._roadsSeperationByDirection(roadsConnected) 
            roadClustersByDirection[road + 1] = roadset_clustered
        return roadClustersByDirection
        
    def runInitialization(self):
        # USE: Calculate the weight of each "placement"
        #     Each "placement" is the combination of a road segment and a direction,
        #     representing the placement of a camera
        # USE: run the simulation and record all the data for further analysis
        # OUTPUT: recordsByRoads - simulation record on each roads
        #         recordsByIterations - simulation record by each simulation iteration, each iteration contains routes of each vehicle
        #         roadClustersByDirection - road cluster information for each road

        # Record all information in the simulation of each road segment / of each iteration
        recordsByRoads = {k + 1: [] for k in range(self.roadData.shape[0])} # Output: the showing up of vehicel on each road 
        recordsByIterations = {k + 1: None for k in range(self.runNum)} # Output: route record of all vehicles of all iterations
        for iter_0 in range(self.runNum):
            routesOnOrigins = {k: None for k in self.startPoints}
            for origin in self.startPoints: 
                routesOfVehicles = {k + 1: None for k in range(int(self.origins[origin - 1]))}
                for vehicle_0 in range(int(self.origins[origin - 1])):
                    route = self._vehicleRoute_toPredictedLocations(origin)
                    routesOfVehicles[vehicle_0 + 1] = route
                    for start, end in zip(route[:-1], route[1:]):
                        detection = {'iteration': iter_0 + 1, 'origin': origin, 'sequence on origin': vehicle_0 + 1, 'step1': start, 'step2': end}
                        recordsByRoads[start].append(detection)
                        recordsByRoads[end].append(detection)
                routesOnOrigins[origin] = routesOfVehicles
            recordsByIterations[iter_0 + 1] = routesOnOrigins    
        
        # Seperating the connected roads to two direction for each road
        roadClustersByDirection = self.getRoadSeperation()

        return recordsByRoads, recordsByIterations, roadClustersByDirection
    
    def _delete_judge (self, delete_list, test):
        iteration = test[0]
        origin = test[1]
        vehicle = test[2]
        captured_vehicle = [[i['origin'], i['vehicle_num']] for i in delete_list if i['iteration_num'] == iteration]
        if [origin, vehicle] in captured_vehicle:
            return True
        else:
            return False
        
    def placementElemination(self, placement_weight_sorted, recordsByRoads, recordsByIterations, withEffectRatio = False):
        # USE: Eliminate the contribution of a placed camera
        #      Return a data without the contribution for calculating the best placement in the rest
        # OUTPUT: updated version of recordsByRoads

        # Get target movements from the placement
        inflow = [[i, placement_weight_sorted[0]['Road']] for i in placement_weight_sorted[0]['Inflow roads']]
        outflow = [[placement_weight_sorted[0]['Road'], i] for i in placement_weight_sorted[0]['Outflow roads']]
        valid_flows = inflow + outflow
        
        # Check if any of the movements exist in recordsByIterations
        # If so, get the index of simulation iteration and the index of suspect vehicle
        # Input: valid_flows
        # Output: delete_list
        delete_list = []
        for iteration_index_0 in range(len(recordsByIterations)):
            reco_iteration = recordsByIterations[iteration_index_0 + 1]
            for origin in list(reco_iteration.keys()):
                for vehicle, get_route in zip(list(reco_iteration[origin].keys()), list(reco_iteration[origin].values())):
                    # judge if the route contains the valid flows indicated by the placement plan
                    for flow in valid_flows:  
                        if flow[0] in get_route:
                            next_step_index = [index for index, x in enumerate(get_route) if x == flow[0]]
                            next_step = []
                            for k in next_step_index:
                                if k == len(get_route) - 1:
                                    continue
                                else:
                                    next_step.append(get_route[k + 1])
                            if flow[1] in next_step:
                                if withEffectRatio == False:
                                    delete_list.append({'iteration_num': iteration_index_0 + 1, 'origin': origin, 'vehicle_num': vehicle})
                                
                                # if consider pre-located cameras
                                elif withEffectRatio == True:
                                    if random.random() <= placement_weight_sorted[0]['EffectRatio']:
                                        delete_list.append({'iteration_num': iteration_index_0 + 1, 'origin': origin, 'vehicle_num': vehicle})
                                        
        # a vehicel from a origin in a iteration may travel through multiple valid flows
        delete_list_short = []
        [delete_list_short.append(x) for x in delete_list if x not in delete_list_short]
        
        global delete_list_short_length_ave
        delete_list_short_length_ave = len(delete_list_short) / self.runNum
        
        # According to the indexes obtained, delete corresponding items from recordsByIterations
        for delete in delete_list_short:
            del recordsByIterations[delete['iteration_num']][delete['origin']][delete['vehicle_num']]
        # According to the indexes obtained, delete corresponding items from recordsByRoads
        updated_recordsByRoads = {k + 1: [] for k in range(self.roadData.shape[0])}
        for road_index_0 in range(len(recordsByRoads)):
            updated_road_reco = [i for i in recordsByRoads[road_index_0 + 1] \
                                 if self._delete_judge(delete_list_short, [i['iteration'], i['origin'], i['sequence on origin']]) == False]
            updated_recordsByRoads[road_index_0 + 1] = updated_road_reco
        return updated_recordsByRoads, recordsByIterations

    def placeMultipleCamera(self, required_camera, preLocated_cameras = None):
        # preLocated_cameras: a list of dict, each dict has "road" and "effect_ratio" keys
        
        # Initialization, run simualtion, and get the first placement of camera
        layout = {k + 1: [] for k in range(required_camera)}
        recordsByRoads, recordsByIterations, roadset_clustered_list = self.runInitialization()
        
        # Consider the existing cameras (virtual cam due to residule effect or fixed one)
        if preLocated_cameras != None: 
            global layout_forPreLocated
            layout_forPreLocated = {k + 1: [] for k in range(len(preLocated_cameras))}
            for k in range(len(preLocated_cameras)):
                preLocated_camera = [preLocated_cameras[k]]
                recordsByRoads, recordsByIterations = \
                    self.placementElemination(preLocated_camera, recordsByRoads, recordsByIterations, withEffectRatio = True)
                layout_forPreLocated[k + 1] = {'Road': preLocated_cameras[k]['Road'], 
                                               'Inflow roads': preLocated_cameras[k]['Inflow roads'], 
                                               'Outflow roads': preLocated_cameras[k]['Outflow roads'], 
                                               'Number of detected vehicles': delete_list_short_length_ave}
        
        # First cam
        placement_weight_sorted = self._placementWeight(recordsByRoads, roadset_clustered_list)
        layout[1] = placement_weight_sorted[0]
        
        # Generating multiple camera placement
        updated_recordsByRoads = recordsByRoads
        updated_recordsByIterations = recordsByIterations
        for i in range(required_camera - 1):
            # Eliminating contribution (of last camera)
            updated_recordsByRoads, updated_recordsByIterations = \
                self.placementElemination(placement_weight_sorted, updated_recordsByRoads, updated_recordsByIterations)
            # Recalculate weights
            placement_weight_sorted = self._placementWeight(updated_recordsByRoads, roadset_clustered_list)
            # Add placement
            layout[i + 2] = placement_weight_sorted[0]
        return layout
       

    ################################################################################
    ########################### Visulization #######################################
    def _get_orientation(self, cor_1, cor_2):
        # INPUT: two lists of [lat,lon]
        origin_y = cor_1[0]
        origin_x = cor_1[1]
        destination_y = cor_2[0]
        destination_x = cor_2[1]    
        deltaX = destination_x - origin_x
        deltaY = destination_y - origin_y
        degrees_temp = math.atan2(deltaX, deltaY)/ math.pi*180
        if degrees_temp < 0:
            degrees_final = 360 + degrees_temp
        else:
            degrees_final = degrees_temp
        compass_brackets = ["North", "Northeast", "East", "Southeast", "South", "Southwest", "West", "Northwest", "North"]
        compass_lookup = round(degrees_final / 45)
        return compass_brackets[compass_lookup], degrees_final
    
    def placementVisualizationMapbox(self, layout):
        # Draw polygons
#         geo_df_select = self.polygon.loc[self.polygon.Prediction > 0] # only draw polygon with predicted crime
        geo_df_select = self.polygon
        geo_df_select['Prediction'] = geo_df_select['Prediction'].astype(str)
        fig = px.choropleth_mapbox(geo_df_select, 
                                   geojson = geo_df_select.geometry,
                                   locations = geo_df_select.index, 
                                   color = "Prediction", 
                                   color_discrete_sequence = px.colors.sequential.Reds,
                                   center = {"lat": 32.606827, "lon": -83.660198},
                                   zoom = 11.8, 
                                   opacity = 0.6, 
                                   hover_data = ['Prediction'],
                                  )  
        fig.update_traces(hovertemplate = "<b>Number of Predicted Crime:</b> %{customdata} <extra></extra>")            
        # Calculate orientation
        road_list = [item['Road'] for item in layout.values()]
        road_out_list = [item['Outflow roads'] for item in layout.values()]
        direction_dict = dict.fromkeys(road_list, None)
        for origin, destination in zip(road_list, road_out_list):
            point1 = self.roadData.loc[self.roadData['OBJECTID'] == origin].midpoint
            road1 = self.roadData.loc[self.roadData['OBJECTID'] == origin].geometry.reset_index()
            road2 = self.roadData.loc[self.roadData['OBJECTID'] == destination[0]].geometry.reset_index()
            point2 = road1.intersection(road2)
            cor1 = [point1.y.values[0],point1.x.values[0]]
            cor2 = [point2.y.values[0],point2.x.values[0]]
            orientation, _ = self._get_orientation(cor1, cor2)
            if direction_dict[origin] == None:
                direction_dict[origin] = orientation
            else:
                direction_dict[origin] = direction_dict[origin] + ', ' + orientation
        # Prepare df
        points_selected = self.roadData.loc[self.roadData['OBJECTID'].isin(road_list)].copy()
        points_selected_index = points_selected.OBJECTID.tolist()
        direction_list = [direction_dict[road] for road in points_selected_index]
        points_selected['direction'] = direction_list
        # Draw scatter plot
        fig2 = px.scatter_mapbox(points_selected, 
                                 lat = points_selected.midpoint.y, 
                                 lon = points_selected.midpoint.x,
                                 opacity = 0.7, 
                                 hover_data = [points_selected.midpoint.y, points_selected.midpoint.x, points_selected.direction])
        fig2.update_traces(marker = dict(size = 10, color = '#D30000'),
                           hovertemplate = "<b>Lat:</b> %{customdata[0]}<br> <b>Lon:</b> %{customdata[1]}<br> <b>Direction:</b> %{customdata[2]} <extra></extra> ")
        fig.add_trace(fig2.data[0])
        fig.update_layout(mapbox_style = "dark", mapbox_accesstoken = self.mapboxKey, margin = {"r": 0, "t": 0, "l": 0,"b": 0}, )
        return fig
        
        
def getPopulationonRoads(population, polygon):
    populationLabelled = population.sjoin(polygon, predicate = 'within')
    populationLabelled = populationLabelled.loc[:, ['#people', 'OBJECTID', 'geometry']]
    polygonLabelled = populationLabelled.dissolve(by = 'OBJECTID', aggfunc = 'sum')
    return polygonLabelled 

def connectLine(pf_road, manage_disconnection = False):
    # USE: Connect the Linestring objects in each MultiLineString objects
    #      Some MultiLineString objects contains LineString objects that are not connected to each other
    #      This function can fix the connecttion problem
    #INPUT: GeoPandas object read from shapefile
    #OUTPUT: Edited GeoPandas object
    idd = 0
    for gg in pf_road.geometry:
        # use the API to merge
        if str(type(gg)) == "<class 'shapely.geometry.multilinestring.MultiLineString'>":
            gg = ops.linemerge(gg)  
        pf_road.geometry[idd] = gg  
        # manage the slighted disconnected ones
        if manage_disconnection == True:
            if str(type(gg)) == "<class 'shapely.geometry.multilinestring.MultiLineString'>":
                gg_decom = [tuple(x.coords) for x in list(gg)]
                gg_c = -1
                for gg_p in gg:
                    gg_c += 1
                    if gg_p.boundary.is_empty:
                        del gg_decom[gg_c]

                gg_com = MultiLineString(gg_decom) 
                for gg1, gg2 in zip(gg_com, gg_com[1:]):
                    newline = LineString([Point(gg1.boundary[1].x, gg1.boundary[1].y),Point(gg2.boundary[0].x, gg2.boundary[0].y)])
                    gg_decom = gg_decom + [tuple(newline.coords)]

                gg = MultiLineString(gg_decom)
                gg = ops.linemerge(gg)
                pf_road.geometry[idd] = gg
        idd += 1
    return pf_road

def roads2Network(pf_input, roadWithPop, buffer = 0):
    #USE: Transform geological road network to topological network
    #INPUT: pf_input: GeoPandas object read from shapefile
    #       buffer: distances below the buffer lead to a connection between lines
    #OUTPUT: NetworkX graph object
    G = nx.Graph()
    i = 1
    for roadLength in pf_input.to_crs("EPSG:3035").length:
        G.add_node(i, length = roadLength)
        # set population
        try:
            G.nodes[i]['pop'] = roadWithPop['#people'].loc[i]
        except: # in case there is no people on this road/polygon
            G.nodes[i]['pop'] = 0
        i += 1
    for obid_1, road_1, length_1 in zip(pf_input.OBJECTID, pf_input.geometry, pf_input.to_crs("EPSG:3035").length):
        for obid_2, road_2, length_2 in zip(pf_input.OBJECTID, pf_input.geometry, pf_input.to_crs("EPSG:3035").length):
            if (obid_1 != obid_2) and road_1.distance(road_2) <= buffer:
                G.add_edge(obid_1, obid_2, weight = (length_1 + length_2) / 2)
            if (obid_1 != obid_2) and (road_1.distance(road_2) < buffer) and (road_1.distance(road_2) > 0):
                print(obid_1, obid_2)
    return G

def degreeCheck(graph):
    #USE: Check the number of lines connecting to nodes
    #INPUT: Networkx graph object
    #OUTPUT: Node degree report
    count = [0, 0, 0]
    for i in range(graph.number_of_nodes()):
        print('Node number: ', i + 1, '; Node degree: ', graph.degree[i + 1])
        if graph.degree[i + 1] < 8:
            count[0] += 1
        elif graph.degree[i + 1] == 8:
            count[1] += 1
        else:
            count[2] += 1
    print('Number of nodes with degree below 8: ', count[0])
    print('Number of nodes with degree equals 8: ', count[1])
    print('Number of nodes with degree above 8: ', count[2]) 