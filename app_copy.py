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

import data_preprocess as dp
import train
import predict_evaluation as pe
import simulation as sim


# Parameters and base map
SelectType = None
TypeColumn = 'Type'
mapbox_access_token = "pk.eyJ1IjoicHhpeW9oIiwiYSI6ImNsMGoxa3h1bzA4ZHQzaW41NWd6dm16am0ifQ.QywfLC6Ut-EhSZLt7nirqQ"
null_data = pd.read_csv("./data/for_make_empty_map.csv")
fig = px.scatter_mapbox(null_data, lat="lat", lon="lon", color_discrete_sequence=["red"],
                       center = {"lat": 32.606827, "lon": -83.660198},
                       zoom = 12, opacity = 0)
fig_initial = fig.update_layout(mapbox_style = "dark", mapbox_accesstoken = mapbox_access_token, margin = {"r":0,"t":0,"l":0,"b":0}, showlegend=False, hovermode=False)
global flag_predict
flag_predict = False
flag_simulation = False


# Web layout
app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}],
)
app.title = "Demo"
server = app.server

# Layout of Dash App
app.layout = html.Div(
    className="row",
    children=[
        # Column for user controls
        html.Div(
            className="four columns div-user-controls",
            children=[ 
                html.Img(
                    className="logo",
                    src=app.get_asset_url("gatech-logo.png"),
                    style={'height':'30%', 'width':'30%'},
                ),
                html.H1("License Plate Recognition Camera Placement APP"),
                html.P(
                    """Input crime history for crime locations prediction and determining the 
                    optimal placements of LPR cameras."""
                ),
                html.P(
                    """* Crime type selection is optional. All data is used if the selector is closed. 
                    Please close the selector if NO specific type or ALL types are interested."""
                ),  
                html.P(
                    """* Press CAMERA PLACEMENT button after finishing CRIME PREDICTION."""
                ),        
                
                # Data upload
                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Files')
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
                html.Div(id='output-data-upload'),
                dcc.Checklist(
                    id = 'crime_type_list', 
                    options = [],
                    value = [],
                    inline = True,
                    style = {
                        'margin-top': '10px',
                        'margin-left': '13px',
                        'margin-right': '10px',
                        'margin-bottom': '0px', 
                }),
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
                    style = {'width': '50%', 'margin': '0px', 'display': 'inline-block'},
                    children = [
                        dcc.Loading(
                            id = 'Loading_1', type = 'circle', color = '#B3A369', 
                            style = {'height': '50%', 'margin-top': '25px', 'margin-left': '30px',},
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
                    style = {'width': '50%', 'margin': '0px', 'display': 'inline-block'},
                    children = [dcc.Loading(id = 'Loading_2', type = 'circle', color = '#B3A369', 
                        style = {'height': '50%', 'margin-top': '25px', 'margin-left': '20px',},
                        children = [                        
                            html.Button('Camera Placement', id='btn-nclicks-2', n_clicks=0,
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
                # Confirm dialog
                dcc.ConfirmDialog(
                    id='confirm-lowData',
                    message='Selected data volumn is too low. Results may be inaccurate. Reselect crime types?',
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
            id = 'map',
            className="eight columns div-for-charts bg-grey",
            children=[
                dcc.Graph(id="map-graph", figure = fig_initial)
            ],
        ),                  
        html.Div(
            id = 'map-temp',
            className = "eight columns div-for-charts bg-grey",
            children = []
        ),                
    ],
)


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    except:
        try:
            df = pd.read_excel(io.BytesIO(decoded))
        except Exception as e:
            print(e)
            return html.Div([
                'There was an error processing this file.'
            ])
    return df
    
def predict(df, SelectType, mapbox_token = mapbox_access_token):
    global pf
    if SelectType == None:
        # Import data, including crime records and polygon shapefile    
        df = dp.Data_import(df)
        # Do resampling
        dfSampled = dp.Data_resampler(df)
        print(dfSampled)
        # Build sequences and datasets
        TrainX, TrainY, ValX, ValY, TestX, TestY = dp.Load_data(dfSampled = dfSampled)
    else:
        df, dfSelected, dfOthers = dp.Data_import(df, SelectType = SelectType)
        dfSelectedSampled, dfOthersSampled = dp.Data_resampler(df, SelectType = SelectType, dfSelected = dfSelected, dfOthers = dfOthers)
        TrainX, TrainY, ValX, ValY, TestX, TestY = dp.Load_data(SelectType = SelectType, \
            dfSelectedSampled = dfSelectedSampled, dfOthersSampled = dfOthersSampled)
    # Train
    model = train.train(TrainX, TrainY, ValX, ValY)
    # Predict
    randomloc = np.random.choice(range(TestX.shape[0]), size=1)[0]
    pf_road = gpd.read_file("./data/MajorRoads.shp").set_crs("EPSG:2240").to_crs("EPSG:4326")
    pf = gpd.read_file("./data/ThiessenPolygons.shp").set_crs("EPSG:2240").to_crs("EPSG:4326")
    BiTestY, BiPredictY, _, _ = pe.CrimePrediction(TestX, TestY, randomloc, model)
    zeroloc, _ = dp.FindEmpty (df, size_resample = '1w')
    pf, BiPredictY_v = pe.CrimeMap(pf, pf_road, BiTestY, BiPredictY, zeroloc, draw = False, get_update_BiPredictY = True)
    fig = pe.CrimeMap_Mapbox(pf, mapbox_token)
    return fig, BiPredictY_v

@app.callback(Output('map', 'children'), 
    Output('btn-nclicks-1', 'type'), 
    Input('upload-data', 'contents'), 
    Input('upload-data', 'filename'),
    Input('btn-nclicks-1', 'n_clicks'), 
    Input('crime_type_list', 'value'),
    Input('confirm-lowData', 'submit_n_clicks'),
    Input('map-temp', 'children'),
    State('crime_type_list', 'options'))
def run_prediction(contents, names, click1, crime_type, lowData_confirm, link, crime_option_state, SelectType = SelectType):
    global flag_predict
    global crime_loc
    if link != []:
        return link, dash.no_update  
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if (contents is not None) and ('btn-nclicks-1' in changed_id):
            df = parse_contents(contents, names)
            if crime_type != []:
                if set(crime_type) != set([item['value'] for item in crime_option_state]):
                    SelectType = crime_type
            print('Model training and crime prediction begins.')
            crime_loc_map, crime_loc = predict(df, SelectType, mapbox_token = mapbox_access_token)
            print('Model training and crime prediction ends.')
            flag_predict = True
            return dcc.Graph(id = "map-graph", figure = crime_loc_map), dash.no_update        
    return dash.no_update, dash.no_update

@app.callback(Output('map-temp', 'children'), 
    Output('btn-nclicks-2', 'type'),
    Input('btn-nclicks-2', 'n_clicks'))
def run_simulation(click2):
    global flag_simulation
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if ('btn-nclicks-2' in changed_id) and (flag_predict == True):
        print('Simulation begins.')
        # Prepare start points and graph
        start_points = []
        loc = 0
        for item in crime_loc:
            loc += 1
            if item == 1:
                start_points.append(loc)
        # Simulation and probability rank
        layout = sim.Placement_weight_forMultiple(start_points, required_camera = 10)
        print(layout)
        # Visualization
        camera_loc_map = sim.Placement_visualization_Mapbox(layout, mapbox_access_token, pf)        
        flag_simulation = True
        print('Simulation ends.')
        return dcc.Graph(id = "map-graph", figure = camera_loc_map), dash.no_update
    return dash.no_update, dash.no_update

@app.callback(Output("output-data-upload", "children"),[
                   Input('upload-data', 'contents'), 
                   Input('upload-data', 'filename'),
              ])
def show_input(contents, names):
    if contents is not None:
        children = [
            html.Div([
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
                html.Button('Type Selector', id='type-selector', n_clicks=0, 
                    style = {
                        'width': '30%',
                        'height': '15px',
                        'lineHeight': 'normal',
                        'borderWidth': '0px',
                        'borderRadius': '2px',
                        'borderColor': '#003057',
                        'textAlign': 'center',
                        'color': '#d8d8d8',
                        'fontFamily': "Open Sans",
                        'font-weight': 'normal',
                        'font-size': '12px',
                        'background-color': '#003057',
                        'margin': '0px',
                        'margin-top': '10px',
                        'display': 'inline-block',
                    }
                ), 
            ])
        ]
        return children
    else:
        children = []  
        return children

@app.callback(Output('crime_type_list', 'options'),
              [Input('upload-data', 'contents'), 
               Input('upload-data', 'filename'),
               Input('output-data-upload', 'n_clicks'),], 
               State('crime_type_list', 'options'))
def show_type(contents, names, click_type, state, typeColumn = TypeColumn):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if (contents is not None) and ('output-data-upload' in changed_id):
        if state == []:
            df = parse_contents(contents, names)
            dff = df.groupby(typeColumn) \
                .size() \
                .reset_index(name='count') \
                .sort_values(['count'], ascending = False) \
                .reset_index(drop=True)
            type_list = [{'label': str(x) + ' (' + str(y) + ')', 'value': str(x)} \
                for x,y in zip(dff['Type'], dff['count'])] 
            return type_list
        elif state != []:
            type_list = []
            return type_list           
    else:
        type_list = []
        return type_list

@app.callback(Output('confirm-lowData', 'displayed'),
              Input('crime_type_list', 'value'),
              Input('upload-data', 'contents'), 
              Input('upload-data', 'filename'),
              Input('btn-nclicks-1', 'n_clicks'))
def show_danger(select_panel, contents, names, click, typeColumn = TypeColumn):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    # if (select_panel != []) and ('btn-nclicks-1' in changed_id):
    if (contents is not None) and (select_panel != []) and ('btn-nclicks-1' in changed_id):
        df = parse_contents(contents, names)
        dff = df.groupby(typeColumn) \
            .size() \
            .reset_index(name='count') \
            .sort_values(['count'], ascending = False) \
            .reset_index(drop=True)
        count_list = [int(dff.loc[dff['Type'] == x, 'count']) for x in select_panel]
        count_sum = sum(count_list)
        if count_sum < 10000:
            return True
    return False 
    
    
   
if __name__ == "__main__":
    app.run_server(debug = True)
