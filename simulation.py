import random
import math
import geopandas as gpd
import plotly.express as px
import networkx as nx
from shapely import ops
from shapely.geometry import Point




def ConnectLine(pf_input, manage_disconnection = False):
    # USE: Connect the Linestring objects in each MultiLineString objects
    #      Some MultiLineString objects contains LineString objects that are not connected to each other
    #      This function can fix the connecttion problem
    #INPUT: GeoPandas object read from shapefile
    #OUTPUT: Edited GeoPandas object

    pf_road = pf_input
    idd = 0
    for gg in pf_road.geometry:
        if str(type(gg)) == "<class 'shapely.geometry.multilinestring.MultiLineString'>":
            gg = ops.linemerge(gg)  
        pf_road.geometry[idd] = gg  
        
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

def Roads2Network(pf_input, buffer = 0):
    #USE: Transform geological road network to topological network
    #INPUT: pf_input: GeoPandas object read from shapefile
    #       buffer: distances below the buffer lead to a connection between lines
    #OUTPUT: NetworkX graph object
    G = nx.DiGraph()
    i = 1
    for road in pf_input.geometry:
        G.add_node(i)
        i += 1
    for obid_1, road_1 in zip(pf_input.OBJECTID, pf_input.geometry):
        for obid_2, road_2 in zip(pf_input.OBJECTID, pf_input.geometry):
            if (obid_1 != obid_2) and road_1.distance(road_2) <= buffer:
                G.add_edge(obid_1, obid_2)
            if (obid_1 != obid_2) and (road_1.distance(road_2) < buffer) and (road_1.distance(road_2) > 0):
                print(obid_1, obid_2)
    return G

def DegreeCheck(graph):
    #USE: Check the number of lines connecting to nodes
    #INPUT: Networkx graph object
    #OUTPUT: Node degree report
    count = [0, 0, 0]
    for i in range(graph.number_of_nodes()):
        print('Node number: ', i + 1, '; Node degree: ', graph.degree[i + 1])
        if graph.degree[i + 1] < 8:
            #print('Node number: ', i + 1, '; Node degree: ', graph.degree[i + 1])
            count[0] += 1
        elif graph.degree[i + 1] == 8:
            count[1] += 1
        else:
            count[2] += 1
    print('Number of nodes with degree below 8: ', count[0])
    print('Number of nodes with degree equals 8: ', count[1])
    print('Number of nodes with degree above 8: ', count[2])   
    

def vehicle_move(graph, node_num):
    #OUTPUT: integer, updated location in network (node)
    neighbor_nodes = list(nx.neighbors(G, node_num))
    next_node = random.randint(0,len(neighbor_nodes) - 1)
    node_num = neighbor_nodes[next_node]
    return node_num

def vehicle_route(graph, node_num, route_len):
    #OUTPUT: list of nodes
    route = [node_num]
    for st in range(route_len):
        node_num = vehicle_move(graph, node_num)
        route.append(node_num)
    return route



def intersection_weight(graph, start_points, route_len):
    #USE: Calcualte times of passing by for each intersection
    #     A vehicle could be counted for several times in each intersection
    #INPUT: graph - NetworkX object with "weight" attribute for every edge
    #       start_points - A list of predicted crime locations
    #OUTPUT: Updated graph
    for point in start_points: 
        route = vehicle_route(G,point,route_len)
        for start, end in zip(route[:-1], route[1:]):
            weight = graph[start][end]['weight'] + 1
            attrs = {(start, end): {"weight": weight}}
            nx.set_edge_attributes(graph, attrs)
    return graph

def movement_weight(graph, start_points, route_len = 50, simu_num = 1000):
    #USE: Calculate the # of suspect vehicles on each "movement"
    #     Each "movement" is a pair of road segment, represented by [x, y]
    #     A same vehicle will not be counted twice on same movement
    #INPUT: graph - NetworkX object with "weight" attribute for every edge
    #OUTPUT: Updated graph
    for simu_time in range(simu_num):
        # print('Number of simulation:', simu_time)
        detection_list = []
        detection_location_list = []
        for point in start_points: 
            route = vehicle_route(G,point,route_len)
            for start, end in zip(route[:-1], route[1:]):
                detection = [point, start, end]

                if detection not in detection_list:
                    weight = graph[start][end]['weight'] + 1
                    attrs = {(start, end): {"weight": weight}}
                    nx.set_edge_attributes(graph, attrs)
                    detection_list.append(detection) 

        for item in [i[1:] for i in detection_list]:
            if item not in detection_location_list:
                detection_location_list.append(item)
        for itemm in detection_location_list:
            st = itemm[0]
            en = itemm[1]
            ifdetect = graph[st][en]['ifdetect'] + 1
            attrs_ifdetect = {(st, en): {"ifdetect": ifdetect}}
            nx.set_edge_attributes(graph, attrs_ifdetect)         
    return graph

def Visual_forMovement(topnum):
    edge_list_selected = edge_list_ranked[0:topnum]
    base_pred = pf.plot(column='Prediction', cmap='Blues', missing_kwds={'color': 'white'}, figsize=(60, 40))
    top_location_map = pf_road.plot(ax=base_pred, color = 'lightgrey')

    for road_pair in edge_list_selected:
        probability_detect = road_pair[0]
        average_detect_num = road_pair[1]
        start_edge = road_pair[2]
        end_edge = road_pair[3]
        top_location_map = pf_road[pf_road.OBJECTID == start_edge].plot(ax=top_location_map, color = 'red', alpha = 0.3)
        top_location_map = pf_road[pf_road.OBJECTID == end_edge].plot(ax=top_location_map, color = 'red', alpha = 0.3)
    top_location_map.set_axis_off()
  



def Placement_road_clustering(road_dict, buffer = 0, lost_warning = False):
    get_line = pf_road[pf_road.OBJECTID == list(road_dict.keys())[0]].geometry.values[0]
    try:
        coords = get_line.coords
    except: # When lines are disconnected, data problem
        coords = [get_line[0].coords[0], get_line[-1].coords[-1]]
        if lost_warning == True:
            print("Remedial measures are taken to extract end points from road segments. Please note there are problems (i.e., lines are not connected in the same group) in the data, although they are solved for this time.")
    c1 = {'v1': []}
    c2 = {'v2': []}  
    for road in list(road_dict.values())[0]:
        # print('assign road', road)
        road_line = pf_road[pf_road.OBJECTID == road].geometry.values[0]
        # print('to v1', road_line.distance(Point(coords[0])))
        # print('to v2', road_line.distance(Point(coords[-1])))
        if road_line.distance(Point(coords[0])) <= buffer:
            c1['v1'].append(road)
        elif road_line.distance(Point(coords[-1])) <= buffer:
            c2['v2'].append(road)
        else:
            if lost_warning == True:
                print('Road', road, 'is not attached to any vetice of the target road', list(road_dict.keys())[0], '.')
    return {list(road_dict.keys())[0]: [c1, c2]}

def Placement_dist_to_crime(road, crimes_start_points):
    mid_point_road = pf_road[pf_road.OBJECTID == road].apply(lambda row: row['geometry'].centroid, axis=1)
    total_dist = 0
    for crime in crimes_start_points:
        mid_point_crime = pf_road[pf_road.OBJECTID == crime].apply(lambda row: row['geometry'].centroid, axis=1)
        # dist = list(mid_point_road)[0].distance(list(mid_point_crime)[0])
        dist = 0
        total_dist += dist
    return total_dist / len(start_points)

def Placement_weight_calculation(simu_record, roadset_clustered_list, simu_num, start_points = None, detection_weight = 1):
    # USE: Calculate the weight of each placement using simulation records and road cluster information
    # INPUT: simu_record - all simulation record grouped by roads
    #        roadset_clustered_list - road cluster information for each road    
    
    placement_weight = []
    for i in range(len(simu_record)):
        record = simu_record[i + 1] 
        roadset_clustered = roadset_clustered_list[i]
        # Calculate pobability of detecting suspect vehicle (# of time of detecting / total simulation times) for each cluster
        roads_v1 = list(roadset_clustered.values())[0][0]['v1']
        roads_v2 = list(roadset_clustered.values())[0][1]['v2']
        effective_simu_time_v1 = list(set([i[0] for i in record if (i[2] in roads_v1 or i[3] in roads_v2)]))
        effective_simu_time_v2 = list(set([i[0] for i in record if (i[2] in roads_v2 or i[3] in roads_v1)]))
        p_v1 = len(effective_simu_time_v1) / simu_num
        p_v2 = len(effective_simu_time_v2) / simu_num
        roadset_clustered['p_v1'] = p_v1
        roadset_clustered['p_v2'] = p_v2
        # Calculate the effect weighted by deterrence and detection
        detection_v1 = p_v1
        detection_v2 = p_v2
#         deterrence_v1 = Placement_dist_to_crime((i + 1), start_points)
#         deterrence_v2 = Placement_dist_to_crime((i + 1), start_points)
        deterrence_v1 = 0
        deterrence_v2 = 0      
        weighted_effect_v1 = detection_weight * detection_v1 + (1 - detection_weight) * deterrence_v1
        weighted_effect_v2 = detection_weight * detection_v2 + (1 - detection_weight) * deterrence_v2
        # Calculate average detection number for each cluster   
        detection_num_v1_temp = [i[0: 2] for i in record if (i[2] in roads_v1 or i[3] in roads_v1)]
        detection_num_v2_temp = [i[0: 2] for i in record if (i[2] in roads_v2 or i[3] in roads_v2)]
        detection_num_v1 = set(tuple(i) for i in detection_num_v1_temp)
        detection_num_v2 = set(tuple(i) for i in detection_num_v2_temp)
        ave_detection_num_v1 = len(detection_num_v1) / simu_num
        ave_detection_num_v2 = len(detection_num_v2) / simu_num
        roadset_clustered['ave_detection_num_v1'] = ave_detection_num_v1
        roadset_clustered['ave_detection_num_v2'] = ave_detection_num_v2
        # Tidy up the data
        out1 = {'Road number':list(roadset_clustered.keys())[0], \
                'Inflow roads':list(roadset_clustered.values())[0][0]['v1'], 'Outflow roads': list(roadset_clustered.values())[0][1]['v2'], \
                'p of detection':list(roadset_clustered.values())[1], 'Ave number of detection':list(roadset_clustered.values())[3], \
                'weighted_effect': weighted_effect_v1}
        out2 = {'Road number':list(roadset_clustered.keys())[0], \
                'Inflow roads':list(roadset_clustered.values())[0][1]['v2'], 'Outflow roads': list(roadset_clustered.values())[0][0]['v1'], \
                'p of detection':list(roadset_clustered.values())[2], 'Ave number of detection':list(roadset_clustered.values())[4], \
                'weighted_effect': weighted_effect_v2}
        placement_weight.append(out1)
        placement_weight.append(out2)
    placement_weight_sorted = sorted(placement_weight, reverse = True, key = take_for_placement)
    return placement_weight_sorted

def Placement_weight_forSingle(start_points, route_len = 25, simu_num = 1000):
    # USE: Calculate the weight of each "placement"
    #     Each "placement" is the combination of a road segment and a direction,
    #     representing the placement of a camera
    # OUTPUT: placement_weight_sorted - the information of all placements
    #         simu_record - simulation record on each roads
    #         routes_records - simulation record by each simulation iteration, each iteration contains routes of each vehicle
    #         roadset_clustered_list - road cluster information for each road
    
    # Record all information in the simulation for each road segment
    simu_record = {k + 1: [] for k in range(pf_road.shape[0])} # Output 1
    routes_records = [] # Output 2
    for simu_time in range(simu_num):
        routes_record = [] 
        for point in start_points: 
            route = vehicle_route(G, point, route_len)
            route_reco = {point: route}
            routes_record.append(route_reco)
            for start, end in zip(route[:-1], route[1:]):
                detection = [simu_time, point, start, end]
                simu_record[start].append(detection)
                simu_record[end].append(detection)
        routes_records.append(routes_record)              
    # Make clusters for each road segment
    roadset_clustered_list = [] # Output
    for i in range(len(simu_record)):
        record = simu_record[i + 1]
        roadset = [i[-2: ] for i in record]
        roadset_temp = list(set(sum(roadset, [])))
        if roadset_temp != []:
            del roadset_temp[roadset_temp.index(i + 1)]
        roadset = {i + 1: roadset_temp}
        roadset_clustered = Placement_road_clustering(roadset) 
        roadset_clustered_list.append(roadset_clustered)
    # Calculate weight
    placement_weight_sorted = Placement_weight_calculation(simu_record, roadset_clustered_list, simu_num, start_points = start_points) # Output
    return placement_weight_sorted, simu_record, routes_records, roadset_clustered_list


def Placement_contri_elimination(placement_weight_sorted, simu_reco, routes_records):
    # USE: Eliminate the contribution of a placed camera
    #      Return a data without the contribution for calculating the best placement in the rest
    # OUTPUT: updated version of simu_reco
    
    # Get target movements from the placement
    inflow = [[i, placement_weight_sorted[0]['Road number']] for i in placement_weight_sorted[0]['Inflow roads']]
    outflow = [[placement_weight_sorted[0]['Road number'], i] for i in placement_weight_sorted[0]['Outflow roads']]
    valid_flows = inflow + outflow
    # Check if any of the movements exist in routes_records
    # If so, get the index of simulation iteration and the index of suspect vehicle
    # Input: valid_flows 
    # Output: delete_list
    delete_list = []
    iteration_index = 0
    for iteration in routes_records:
        for vehicle in iteration:
            vehicle_num = list(vehicle.keys())[0]
            get_route = list(vehicle.values())[0]
            for flow in valid_flows:  
                if flow[0] in get_route:
                    next_step_index = [i for i, x in enumerate(get_route) if x == flow[0]]
                    next_step = []
                    for i in next_step_index:
                        try:
                            next_step.append(get_route[i + 1])
                        except: 
                            if i != (len(get_route) - 1):
                                print('Error. It should always be len(get_route)-1, but it is ', i, ' right now.')
                    if flow[1] in next_step:
                       delete_list.append({'iteration_num': iteration_index, 'vehicle_num': vehicle_num})
        iteration_index += 1
    delete_list_short = []
    [delete_list_short.append(x) for x in delete_list if x not in delete_list_short]
    # According to the indexes obtained, delete corresponding items from simu_reco
    # Input: delete_list
    # Output: updated_simu_reco
    updated_simu_reco = {k + 1: [] for k in range(pf_road.shape[0])}
    i = 1
    for road in list(simu_reco.values()):
        updated_road_reco = [i for i in road if delete_judge(delete_list_short, [i[0], i[1]]) == False]
        updated_simu_reco[i] = updated_road_reco
        i += 1
    return updated_simu_reco

def Placement_weight_forMultiple(start_points, required_camera = 3, route_len = 25, simu_num = 1000):
    # Initialization
    layout = {k + 1: [] for k in range(required_camera)}
    placement_weight_sorted, simu_record, routes_records, roadset_clustered_list = Placement_weight_forSingle(start_points)
    layout[1] = placement_weight_sorted[0]
    # Generating multiple camera placement
    updated_simu_reco = simu_record
    for i in range(required_camera - 1):
        # Eliminating contribution (of last camera)
        updated_simu_reco = Placement_contri_elimination(placement_weight_sorted, updated_simu_reco, routes_records)
        # Recalculate weights
        placement_weight_sorted = Placement_weight_calculation(updated_simu_reco, roadset_clustered_list, simu_num)
        # Add placement
        layout[i + 2] = placement_weight_sorted[0]
    return layout

def Placement_visualization(layout):
    base_pred = pf.plot(column='Prediction', cmap='Blues', missing_kwds={'color': 'white'}, figsize=(60, 40))
    top_location_map = pf_road.plot(ax=base_pred, color = 'lightgrey')
    for i in range(len(layout)):
        placement = layout[i + 1]
        road = placement['Road number']
        inflows = placement['Inflow roads']
        for inflow in inflows:
            pair = [inflow, road]
            top_location_map = pf_road[pf_road.OBJECTID == pair[0]].plot(ax=top_location_map, color = 'red', alpha = 0.6)
            top_location_map = pf_road[pf_road.OBJECTID == pair[1]].plot(ax=top_location_map, color = 'red', alpha = 0.6)
    top_location_map.set_axis_off()

def Placement_visualization_Mapbox(layout, mapbox_token, pf):
    # Draw polygons
    geo_df_select = pf[pf.Prediction == 1]
    fig = px.choropleth_mapbox(geo_df_select, geojson = geo_df_select.geometry,
                               locations = geo_df_select.index, color_discrete_sequence=["red"],
                               center = {"lat": 32.606827, "lon": -83.660198},
                               zoom = 12, opacity = 0.5, hover_data = ['OBJECTID'])
    fig.update_layout(mapbox_style = "dark", mapbox_accesstoken = mapbox_token, margin = {"r":0,"t":0,"l":0,"b":0}, showlegend=False)
    fig.update_traces(hovertemplate = "<b>ID:</b> %{customdata[0]}")
    # Calculate orientation
    road_list = [item['Road number'] for item in layout.values()]
    road_out_list = [item['Outflow roads'] for item in layout.values()]
    direction_dict = dict.fromkeys(road_list, None)
    for origin, destination in zip(road_list, road_out_list):
        point1 = pf_road.loc[pf_road['OBJECTID'] == origin].midpoint
        road1 = pf_road.loc[pf_road['OBJECTID'] == origin].geometry.reset_index()
        road2 = pf_road.loc[pf_road['OBJECTID'] == destination[0]].geometry.reset_index()
        point2 = road1.intersection(road2)
        cor1 = [point1.y.values[0],point1.x.values[0]]
        cor2 = [point2.y.values[0],point2.x.values[0]]
        orientation, _ = get_orientation(cor1, cor2)
        if direction_dict[origin] == None:
            direction_dict[origin] = orientation
        else:
            direction_dict[origin] = direction_dict[origin] + ', ' + orientation
    # Prepare df
    points_selected = pf_road[pf_road['OBJECTID'].isin(road_list)]
    points_selected_index = points_selected.OBJECTID.tolist()
    direction_list = [direction_dict[road] for road in points_selected_index]
    points_selected['direction'] = direction_list
    points_selected['color'] = 'default'
    # Draw scatter plot
    fig2 = px.scatter_mapbox(points_selected, lat=points_selected.midpoint.y, lon=points_selected.midpoint.x,
                             color = 'color', color_discrete_map = {'default': '#B3A369'},
                             center = {"lat": 32.606827, "lon": -83.660198},
                             zoom = 11.5, opacity = 0.7, 
                             hover_data = [points_selected.midpoint.y, points_selected.midpoint.x, points_selected.direction])
    fig2.update_layout(mapbox_style="dark", mapbox_accesstoken=mapbox_token,
                     margin = {"r":0,"t":0,"l":0,"b":0}, showlegend=False)
    fig2.update_traces(marker=dict(size=10),
                      hovertemplate = "<b>Lat:</b> %{customdata[0]}<br> <b>Lon:</b> %{customdata[1]}<br> <b>Direction:</b> %{customdata[2]} <extra></extra> ")
    # Combine two figs
    fig.add_trace(fig2.data[0])
    for i, frame in enumerate(fig.frames):
        fig.frames[i].data += (fig2.frames[i].data[0],)
    return fig


def Placement_visualization_midpoints():
    try:
        pf_road['midpoint']
        base = pf_road['geometry'].plot(figsize=(60, 40))
        pf_road['midpoint'].plot(ax = base)
    except:
        print('Get midpoints for geoDataframe')

def takefirst(item):
    return item[0]

def take_for_placement(item):
    return item['p of detection']

def delete_judge (delete_list, test):
    iteration = test[0]
    vehicle = test[1]
    captured_vehicle = [i['vehicle_num'] for i in delete_list if i['iteration_num'] == iteration]
    if vehicle in captured_vehicle:
        return True
    else:
        return False

def get_orientation(cor_1, cor_2):
    # INPUT: two lists of [lat,lon]
    origin_y = cor_1[0]
    origin_x = cor_1[1]
    destination_y = cor_2[0]
    destination_x = cor_2[1]    
    deltaX = destination_x - origin_x
    deltaY = destination_y - origin_y
    degrees_temp = math.atan2(deltaX, deltaY)/math.pi*180
    if degrees_temp < 0:
        degrees_final = 360 + degrees_temp
    else:
        degrees_final = degrees_temp
    compass_brackets = ["North", "Northeast", "East", "Southeast", "South", "Southwest", "West", "Northwest", "North"]
    compass_lookup = round(degrees_final / 45)
    return compass_brackets[compass_lookup], degrees_final


# Create topological network from road network and add midpoint
pf_road = gpd.read_file("./data/MajorRoads.shp").set_crs("EPSG:2240").to_crs("EPSG:4326")
pf_road.insert(0, 'OBJECTID', range(1, 1 + len(pf_road)))
pf_road = ConnectLine(pf_road)
pf_road_temp = pf_road.to_crs("EPSG:3035")
pf_road['midpoint'] = pf_road_temp['geometry'].centroid.to_crs("EPSG:4326")
G = Roads2Network(pf_road)