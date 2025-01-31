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
            coords = [get_line.geoms[0].coords[0], get_line.geoms[-1].coords[-1]]
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

    def placementVisualizationMapbox(self, layout, timeRange_text = None, show_raw_data = False):
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
        # Layout
        fig.update_layout(mapbox_style = "light",
                          mapbox_accesstoken = self.mapboxKey,
                          title_text = 'LPR Placement Plan and Crime Pattern Prediction' + '  (' + timeRange_text + ')',
                          margin = {"r":30,"l":30,"b":60},
                          autosize = False, width = 800, height = 600,
                          annotations = [dict(text = "Note: LPRs will maintain same theoretical effectiveness placed within the road segment range. <br>Network Dynamics Lab: http://ndl.gatech.edu/",
                                              xref = "paper", yref = "paper",
                                              x = 0, y = -0.08,
                                              showarrow = False,
                                              align = 'left', )],)
        # Combine raw crime location
        if show_raw_data == True:
            fig_raw = self.crimePointMap(self.testWeek)
            fig.add_trace(fig_raw.data[0])
        return fig

    def residuleAnalysisPending(self):
        print(layout_withoutResidual)
        print(layout_withResidual)
        print(layout_forPreLocated)

        layout_withoutResidual_list = [layout_withoutResidual[k + 1]['Number of detected vehicles'] for k in range(len(layout_withoutResidual))]
        layout_withoutResidual_list = [0, 0, 0] + layout_withoutResidual_list
        accumulate_layout_withoutResidual_list = [sum(layout_withoutResidual_list[:i + 1]) for i in range(len(layout_withoutResidual_list))]

        layout_withResidual_list = [layout_withResidual[k + 1]['Number of detected vehicles'] for k in range(len(layout_withResidual))]
        layout_withResidual_list = [0, 0, 0] + layout_withResidual_list
        accumulate_layout_withResidual_list = [sum(layout_withResidual_list[:i + 1]) for i in range(len(layout_withResidual_list))]

        layout_withResidual_withVirtualOnes_list = [layout_withResidual[k + 1]['Number of detected vehicles'] for k in range(len(layout_withResidual))]
        layout_withResidual_withVirtualOnes_list = [layout_forPreLocated[k + 1]['Number of detected vehicles'] for k in range(len(layout_forPreLocated))] \
            + layout_withResidual_withVirtualOnes_list
        accumulate_layout_withResidual_withVirtualOnes_list = [sum(layout_withResidual_withVirtualOnes_list[:i + 1]) \
                                                              for i in range(len(layout_withResidual_withVirtualOnes_list))]

        fig = go.Figure()
        camNum = range(len(accumulate_layout_withResidual_list))
        fig.add_trace(go.Scatter(x = [item + 1 for item in camNum], y = accumulate_layout_withoutResidual_list, fill = 'tozeroy', name = 'Real effect'))
        fig.add_trace(go.Scatter(x = [item + 1 for item in camNum], y = accumulate_layout_withResidual_list, fill = 'tozeroy', name = 'Real effect considering residual effect'))
        fig.add_trace(go.Scatter(x = [item + 1 for item in camNum], y = accumulate_layout_withResidual_withVirtualOnes_list, fill = 'tonexty', name = 'Real effect + Residual effect'))
        fig.update_layout(xaxis_title="Index of cameras", yaxis_title="# of vehicles captured (124 in total)")
        fig.show()
