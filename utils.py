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


