import dash
from dash import dcc
from dash import html
from dash import callback_context
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px

import pandas as pd
import geopandas as gpd
import numpy as np
import base64
import io
from datetime import datetime as dt

import functions
from spektral.utils import gcn_filter


mapbox_access_token = "pk.eyJ1IjoicHhpeW9oIiwiYSI6ImNsMGoxa3h1bzA4ZHQzaW41NWd6dm16am0ifQ.QywfLC6Ut-EhSZLt7nirqQ"
fig_initial = px.scatter_mapbox(pd.DataFrame(data = {'lat': 32.6130007, 'lon': -83.624201}, index = [0, 1]),
                        lat = "lat", lon = "lon", color_discrete_sequence = ["red"],
                        center = {"lat": 32.606827, "lon": -83.660198},
                        mapbox_style = "dark",
                        zoom = 12, opacity = 0
                        )\
                        .update_layout(mapbox_accesstoken = mapbox_access_token, 
                        margin = {"r": 0, "t": 0, "l": 0, "b": 0}, 
                        showlegend = False, 
                        hovermode = False,
                        )


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    readData = functions.dataProcess().dataImport
    try:
        # df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        df = readData(io.StringIO(decoded.decode('utf-8')))
    except:
        try:
            df = readData(io.BytesIO(decoded))
        except Exception as e:
            print(e)
            return html.Div([
                'There was an error processing this file.'
            ])
    return df
    
def predict(df, dayIterval = 7, lenSequence = 24):
    # Import data, including crime records and polygon shapefile
    dp = functions.dataProcess('./data/ThiessenPolygons.shp', 'EPSG:2240')
    dfSampled = dp.dataResampler(df, '168h')
    trainX1, trainY1, trainX2, trainY2, valX, valY, X4Predict = dp.buildDatasets(dfSampled, lenSequence, randomIndex = False)
    trainX = np.concatenate((trainX1, trainX2), axis=0)
    trainY = np.concatenate((trainY1, trainY2), axis=0)
    
    # Initialize crime prediciton
    global cp
    global prediction 
    cp = functions.crimePrediction()
    cp.mapboxKey = mapbox_access_token
    cp.pf = gpd.read_file("./data/ThiessenPolygons.shp").set_crs("EPSG:2240").to_crs("EPSG:4326")
    # cp.pf_road = gpd.read_file("./data/MajorRoads.shp").set_crs("EPSG:2240").to_crs("EPSG:4326")
    cp.size_resample = dp.size_resample
    
    dfs = df.iloc[0 : int(len(df) / 10)]
    weeks = int ((dfs.index[0] - dfs.index[-1]).days / dayIterval)
    crimeMapFull = np.rint(dp.crimeRaster(dfs) / weeks)[0]
    cp.crimeMapFull = crimeMapFull
    
    # get adjMatrix for graph convolution
    adjMatrix = np.load('./data/adjMatrix.npy')
    lapacian = gcn_filter(adjMatrix, symmetric = True)
    lapacian_train = np.tile(lapacian, (trainX.shape[0], 1, 1))
    lapacian_val = np.tile(lapacian, (valX.shape[0], 1, 1))

    # Train
    cp.trainedModel = cp.train(trainX, trainY, valX, valY, lap_train = lapacian_train, lap_val = lapacian_val, \
                            batch_size = 6, epochs = 10000, learning_rate = 2e-05, patienceReduceLR = 5, \
                            alpha_roundApproximation = 5, lowerMultiplier = 0.9) 

    # Predict
    prediction = cp.predict(X4Predict, np.empty(X4Predict.shape), 0, lap = lapacian) 
    cp.polygonUpdate(prediction) 
    return cp.crimePolygonMap()
 


app = dash.Dash(
    __name__, 
    title = "LPR Dynamic Placement APP",
)

app.layout = html.Div(
    children=[
        html.Div(
            className = "row",
            children=[
                # Column for user controls
                html.Div(
                    className = "four columns div-user-controls",
                    children=[
                        # texts
                        html.A(
                            html.Img(
                                className = "logo",
                                src = app.get_asset_url("gatech-logo.png"),
                                style = {'height':'30%', 'width':'30%'},
                            ),
                        ),
                        html.H2("License Plate Recognition Camera Placement APP"),
                        html.P(
                            """Input crime records in .csv format for crime pattern prediction and LPR placement determination."""
                        ),
                        html.P(
                            """* Press CAMERA PLACEMENT button after finishing CRIME PREDICTION."""
                        ),                          
                        # Data upload
                        html.Div(
                            children=[
                                dcc.Upload(
                                    id = 'upload-data',
                                    children = html.Div([
                                        'Drag and Drop or ', html.A('Select Files')
                                    ]),
                                    style={
                                        'width': '95%',
                                        'height': '60px',
                                        'lineHeight': '60px',
                                        'borderWidth': '1px',
                                        'borderStyle': 'dashed',
                                        'borderRadius': '5px',
                                        'textAlign': 'center',
                                        'margin-left': '15px',
                                        'margin-top': '30px',
                                        'margin-bottom': '0px',
                                    },
                                ),
                            ],
                        ),
                        html.Div(id = 'output-data-upload'),
                        html.Hr(
                            style={
                                'margin-top': '10px',
                                'margin-left': '15px',
                                'margin-right': '10px',
                                'margin-bottom': '0px',
                            }
                        ),                         
                        # Start button
                        html.Div(
                            style = {
                                'width': '50%', 
                                'margin': '0px', 
                                'display': 'inline-block'},
                                children = [dcc.Loading(
                                    id = 'Loading_1', 
                                    type = 'circle', 
                                    color = '#B3A369', 
                                    style = {
                                        'height': '50%', 
                                        'margin-top': '25px', 
                                        'margin-left': '30px',},
                                    children = [
                                        html.Button('Crime Prediction', id='btn-nclicks-1', n_clicks=0,
                                            style={
                                              'width': '93%',
                                              'height': '60px',
                                              'lineHeight': '60px',
                                              'borderWidth': '1px',
                                              'borderRadius': '5px',
                                              'borderColor': '#B3A369',
                                              'textAlign': 'center',
                                              'color': '#003057',
                                              'fontFamily': "Open Sans",
                                              'font-weight': 'normal',
                                              'font-size': '14px',
                                              'background-color': '#B3A369',
                                              'margin-top': '10px',
                                              'float': 'left',
                                              'margin-left': '15px',
                                              'margin-bottom': '30px',
                                            }
                                        ),
                                    ]
                                )],  
                        ),
                        html.Div(
                            style = {
                                'width': '50%', 
                                'margin': '0px', 
                                'display': 'inline-block'},
                                children = [dcc.Loading(
                                    id = 'Loading_2', 
                                    type = 'circle', 
                                    color = '#B3A369', 
                                    style = {
                                        'height': '50%', 
                                        'margin-top': '25px', 
                                        'margin-left': '20px',},
                                    children = [
                                        html.Button('Camera Placement', id = 'btn-nclicks-2', n_clicks = 0,
                                            style={
                                                  'width': '93%',
                                                  'height': '60px',
                                                  'lineHeight': '60px',
                                                  'borderWidth': '1px',
                                                  'borderRadius': '5px',
                                                  'borderColor': '#B3A369',
                                                  'textAlign': 'center',
                                                  'color': '#003057',
                                                  'fontFamily': "Open Sans",
                                                  'font-weight': 'normal',
                                                  'font-size': '14px',
                                                  'background-color': '#B3A369',
                                                  'margin-top': '10px',
                                                  'float': 'right',
                                                  'margin-right': '10px',
                                                  'margin-bottom': '30px',
                                                }
                                        ),
                                    ]
                                )],  
                        ),                    
                        # Links
                        dcc.Markdown(
                            """
                            Links: [Network Dynamics Lab](https://ndl.gatech.edu/)
                            """
                        ),
                    ],
                ),
                
                # Column for visualization  
                html.Div(
                    id = 'map-div',
                    className="eight columns div-for-charts bg-grey",
                    children=[
                        dcc.Graph(id = "map-graph", figure = fig_initial)
                    ],
                ),                
            ],
        )
    ]
)



@app.callback(Output("output-data-upload", "children"),[
                Input('upload-data', 'contents'), 
                Input('upload-data', 'filename'),
              ])
def show_input(contents, names):
    if contents is not None:
        return html.Div([
             html.H5(names,
                style={
                    'width': '65%',
                    'font-family': "Open Sans",
                    'font-weight': 'normal',
                    'font-size': '12px',
                    'margin': '0px',
                    'margin-top': '10px',
                    'margin-left': '5px',
                    'display': 'inline-block',
                }
            ),
        ])
    else:
        return []


@app.callback(
    Output('map-div', 'children'), 
    Output('btn-nclicks-1', 'type'), 
    Input('upload-data', 'contents'), 
    Input('upload-data', 'filename'),
    Input('btn-nclicks-1', 'n_clicks'), 
    )
def run_prediction(contents, names, click1):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if (contents is not None) and ('btn-nclicks-1' in changed_id):
    
        print('Model training and crime prediction begins.')
        crime_pattern_map = predict(parse_contents(contents, names))
        print('Model training and crime prediction ends.')
        return dcc.Graph(id = "map-graph", figure = crime_pattern_map), dash.no_update
            
    return dash.no_update, dash.no_update


@app.callback(Output('map-graph', 'figure'), 
    Output('btn-nclicks-2', 'type'),
    Input('btn-nclicks-2', 'n_clicks'))
def run_simulation(click2):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if 'btn-nclicks-2' in changed_id:
        print('Simulation begins.')
        # get graph
        roadData = gpd.read_file("./data/MajorRoads.shp").set_crs("EPSG:2240").to_crs("EPSG:4326")
        roadData.insert(0, 'OBJECTID', range(1, 1 + len(roadData)))
        roadData['length'] = (roadData.to_crs("EPSG:3035"))['geometry'].length # Add road length
        roadData['midpoint'] = (roadData.to_crs("EPSG:3035"))['geometry'].centroid.to_crs("EPSG:4326") # Add midpoints to major road geoDataframe
        population = gpd.read_file("./data/shapefile_populationDist/populationDist.shp")
        polygon = gpd.read_file("./data/ThiessenPolygons.shp").set_crs("EPSG:2240").to_crs("EPSG:4326")
        roadWithPop = functions.getPopulationonRoads(population, polygon)
        roadData = functions.connectLine(roadData)
        G = functions.roads2Network(roadData, roadWithPop)

        # Initialize class
        cp.pf.Prediction = cp.pf.Prediction.astype(int)
        sim = functions.simulation(prediction, roadData, G, 250, 10, polygon = cp.pf, mapboxKey = mapbox_access_token)
        # run
        layout_withResidual = sim.placeMultipleCamera(required_camera = 10,
            # preLocated_cameras = 
                 # [{'Road': 106, 'Inflow roads': [56, 208], 'Outflow roads': [50, 75, 107], 'EffectRatio': 0.3}, 
                  # {'Road': 76, 'Inflow roads': [125, 201], 'Outflow roads': [203, 266], 'EffectRatio': 0.5}, 
                  # {'Road': 38, 'Inflow roads': [41, 249, 258], 'Outflow roads': [262, 274], 'EffectRatio': 0.4}]
        )
        
        print('Simulation ends.')
        return sim.placementVisualizationMapbox(layout_withResidual), dash.no_update
    return dash.no_update, dash.no_update



if __name__ == "__main__":
    app.run_server(debug = True)
