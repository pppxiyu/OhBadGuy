import math
import plotly.express as px
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix , classification_report
from sklearn.metrics import roc_curve, roc_auc_score

def CrimePrediction(TestX, TestY, randomloc, model, \
                    CrimeNumBottom = 15,  CrimeThreshold = 0.7, ThresholdTuneStep = 0.01, TuneSwitch = True):

    TestX_local = TestX[randomloc]
    TestY_local = TestY[randomloc]
    TestY_local = np.squeeze(TestY_local)

    def crimepredict(TestX_local, TestY_local):
        # Predict crime based on crime history
        PredictY = model.predict(np.expand_dims(TestX_local, axis = 0))
        PredictY = np.squeeze(PredictY)

        # Binary Y
        BiTestY = TestY_local
        BiTestY[np.nonzero(BiTestY)] = 1
        BiTestY = BiTestY.tolist()
        BiPredictY = np.around(PredictY, 8)
        YesCrime = np.where(BiPredictY >= ((np.max(BiPredictY) - np.min(BiPredictY)) * CrimeThreshold))
        NoCrime = np.where(BiPredictY < ((np.max(BiPredictY) - np.min(BiPredictY)) * CrimeThreshold))
        BiPredictY[YesCrime] = 1
        BiPredictY[NoCrime] = 0
        BiPredictY = BiPredictY.tolist()
        return BiTestY, BiPredictY, TestY_local, PredictY

    if TuneSwitch == False:
        BiTestY, BiPredictY, TestY_local, PredictY = crimepredict(TestX_local, TestY_local)
    else:
        CrimeNum = 10000000
        CrimeThreshold = CrimeThreshold
        while CrimeNum > CrimeNumBottom:
            BiTestY, BiPredictY, TestY_local, PredictY = crimepredict(TestX_local, TestY_local)
            CrimeNum = sum(BiPredictY)
            CrimeThreshold += ThresholdTuneStep
            
    return BiTestY, BiPredictY, TestY_local, PredictY


       
def BiMetrics(BiTestY, BiPredictY): 
    # Binary confusion matrix
    cm = confusion_matrix(BiTestY, BiPredictY)
    print_confusion_matrix(cm, ['No Crime','Crime'],  fontsize = 14)

    # Binary F1
    print(classification_report(BiTestY, BiPredictY))

    # The expected performamce of crime detection by cameras
    Crime_precision = cm[1][1] / (cm[1][1] + cm[0][1])
    Camera_num = 5
    # The probability of that x in n cameras can detect crime
    def binomial_distribution(p,n,x):
        c=math.factorial(n)/math.factorial(n-x)/math.factorial(x)
        return c*(p**x)*((1-p)**(n-x))
    accu_prob = 0
    for detection_num in np.arange(0,Camera_num + 1):
        detection_prob = binomial_distribution(Crime_precision, Camera_num, detection_num)
        #detection_prob = binomial_distribution(1, Camera_num, detection_num)
        accu_prob = accu_prob + detection_prob
        [print('\n The probability of that %d cameras in %d cameras can detect crime is %.10f' % (detection_num, Camera_num, detection_prob * 100), '%', 
               '\n The probability of that at least %d cameras in %d cameras can detect crime is %.10f' % (detection_num + 1, Camera_num, (1 - accu_prob) * 100), '%')]

        
def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('Truth',  fontsize=fontsize)
    plt.xlabel('Prediction',  fontsize=fontsize)

    
def ROC_plot(TestY, PredictY):
    Crime_fpr, Crime_tpr, Crime_threshold = roc_curve(TestY, PredictY)  
    plt.figure()
    plt.plot(Crime_fpr, Crime_tpr, marker='.')
    plt.title('ROC Plot')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #Randon counterpart
    Random_probs = [0 for _ in range(len(BiTestY))]
    Random_fpr, Random_tpr, Random_threshold = roc_curve(BiTestY, Random_probs)
    plt.plot(Random_fpr, Random_tpr, linestyle='--')
    plt.show()
    
    
def PrecisionPlot(TestX, TestY, model):
    # Precision plot and the plot of number of crime prediction
    Precision_record = []
    NumCrime_record = []
    for loc in range(TestX.shape[0]):

        BiTestY, BiPredictY, _, _ = CrimePrediction(TestX, TestY, loc, model)

        NumCrime = sum(BiPredictY)
#         if NumCrime < CrimeNum_bottom:
#             continue
#         else:   
        NumCrime_record.append(NumCrime)

        clas_report = classification_report(BiTestY, BiPredictY, output_dict=True)
        #print(clas_report)
        precision = clas_report["1.0"]
        precision = precision['precision']
        #print(precision)
        Precision_record.append(precision)

    y = np.asarray(Precision_record)
    x = np.arange(1,len(Precision_record) + 1)
    plt.plot(x,y)
    plt.ylim((0,1))
    plt.title('Precision')
    print('Average of the precision is', sum(Precision_record) / len(Precision_record))

    y = np.asarray(NumCrime_record)
    x = np.arange(1,len(NumCrime_record) + 1)
    plt.figure()
    plt.plot(x,y)
    plt.ylim((0,100))
    plt.title('Number of Crime in Prediction')

def CrimeMap(pf, pfr, BiTestY, BiPredictY, zeroloc, draw = True, get_update_BiPredictY = False):
    BiTestY_v = []
    for Bi in BiTestY:
        BiTestY_v.append(Bi)
    BiPredictY_v = []
    for Bi in BiPredictY:
        BiPredictY_v.append(Bi)
    for loc in zeroloc[1]:
        BiTestY_v.insert(loc, None)
        BiPredictY_v.insert(loc, None)
    BiFalsePositiveY = []
    for BT, BP in zip(BiTestY_v, BiPredictY_v):
        if BP == 1 and BT == 0:
            BiFalsePositiveY.append(1)
        elif BP == None or BT == None:
            BiFalsePositiveY.append(None)
        else:
            BiFalsePositiveY.append(0)

    pf['GroundTruth'] = BiTestY_v
    pf['Prediction'] = BiPredictY_v
    pf['FalsePositive'] = BiFalsePositiveY
    
    if draw == True:
        base1 = pf.plot(column='GroundTruth', cmap='OrRd', missing_kwds={'color': 'lightgrey'}, figsize=(15, 10))
        pfr.plot(ax=base1).set_axis_off()
        base2 = pf.plot(column='Prediction', cmap='OrRd', missing_kwds={'color': 'lightgrey'}, figsize=(15, 10))
        pfr.plot(ax=base2).set_axis_off()
        base3 = pf.plot(column='FalsePositive', cmap='OrRd', missing_kwds={'color': 'lightgrey'}, figsize=(15, 10))
        pfr.plot(ax=base3).set_axis_off() 
    
    if get_update_BiPredictY == True:
        return pf, BiPredictY_v
    
    return pf

def CrimeMap_Mapbox(geo_df, mapbox_token):
    geo_df_select = geo_df[geo_df.Prediction == 1]
    fig = px.choropleth_mapbox(geo_df_select, geojson = geo_df_select.geometry,
                               locations = geo_df_select.index, color_discrete_sequence=["red"],
                               center = {"lat": 32.606827, "lon": -83.660198},
                               zoom = 12, opacity = 0.5)
    fig.update_layout(mapbox_style = "dark", mapbox_accesstoken = mapbox_token, margin = {"r":0,"t":0,"l":0,"b":0}, showlegend=False, hovermode=False)
    return fig

if __name__ == '__main__':
    mapbox_token = 'pk.eyJ1IjoicHhpeW9oIiwiYSI6ImNsMGoxa3h1bzA4ZHQzaW41NWd6dm16am0ifQ.QywfLC6Ut-EhSZLt7nirqQ' 
    randomloc = np.random.choice(range(TestX.shape[0]), size=1)[0]
    pf_road = gpd.read_file("./data/MajorRoads.shp").set_crs("EPSG:2240").to_crs("EPSG:4326")
    pf = gpd.read_file("./data/ThiessenPolygons.shp").set_crs("EPSG:2240").to_crs("EPSG:4326")

    BiTestY, BiPredictY, TestY_one, PredictY = CrimePrediction(TestX, TestY, randomloc, model)
    zeroloc, _ = FindEmpty (df, size_resample = '1w')
    BiPredictY_v = CrimeMap(pf, pf_road, BiTestY, BiPredictY, zeroloc, draw = False, get_update_BiPredictY = True)
    CrimeMap_Mapbox(pf, mapbox_token)

    # BiMetrics(BiTestY, BiPredictY)
    # ROC_plot(TestY_one, PredictY)
    # CrimeNum = PrecisionPlot(TestX, TestY, model)