import os
import pandas as pd
import joblib
import numpy as np
import random
from collections import defaultdict
from sklearn.model_selection import GridSearchCV
from utils.load import load_everthing
from utils.load import load
from utils.load import load_everthing_with_continents
from utils.load import load_everthing_with_distance
from utils.load import load_all_distance_metrics
from utils.load import loader
from utils.plots import plot_confidence_interval

def naming(path: str, name: str, extension: str) -> str:
    return path + name + "_" + str(len(os.listdir(path)) + 1) +"." + extension

def merge_dfs(path = "martin_runs/"):
    df_list = []
    for elm in os.listdir(path):
        if "merged" in elm:
            continue
        df_list.append(pd.read_csv(path + elm))
    df = pd.concat(df_list)
    df.to_csv(f"{path}/merged_df.csv",index = False)

class result_object():
    def __init__(self, grid_search: object, dataset: str,X,Y,region,with_distance, save_model = True):
        self.gridsearch = grid_search
        self.data_type = dataset
        self.model_name = naming("martin_models/","test","pkl")
        self.run_name = naming("martin_runs/","test","csv")
        self.pipeline = grid_search.estimator
        self.region = region
        self.with_distance = with_distance
        if save_model == True:
            joblib.dump(self.gridsearch, self.model_name) 
        d = pd.DataFrame(self.gridsearch.cv_results_)
        d = d[["params","mean_test_score"]]
        d["dataset"] =  [dataset for _ in range(len(d)) ]
        #d["score"] = grid_search.best_score_
        d["model_file_name"] =  [self.model_name for _ in range(len(d)) ]
        d["pipeline"] = [self.pipeline for _ in range(len(d)) ] 
        d["region"] = [self.region for _ in range(len(d))]
        d["with_distance"] = [self.with_distance for _ in range(len(d))]
        d["model_name"] = [self.pipeline[-1] for _ in range(len(d)) ] 
        self.dataframe = d
        self.X = X
        self.Y = Y
        
        
        self.dataframe.to_csv(self.run_name, index = False)




    

def get_params(path):
    """It returns a list of tuples, where the tuples are the featurenames and the coefficient
    the path is the path to the given pkl model"""
    feature_list = []
    for file in os.listdir("data/fb_data/"):
        if ("CosDist" in file) and (file.endswith(".csv")) and (file != "FBCosDist.csv"):
            feature = file.split("_")[1]
            feature = feature.replace(".csv","")
            feature_list.append(feature)
    clf = load(path)
    li = []
    for coef, feature in zip(clf.best_estimator_.steps[-1][-1].coef_, feature_list):
        li.append((feature, coef))
    return li



def get_pred_and_labels(clf,n = 0, with_distance = False, all_distance_metrics = False):
    """samples n amount of datapoints and uses the model clf to predict
    if n = 0, then we dont sample anything and use the entire dataset"""
    if all_distance_metrics:
        print("1")
        print("with_distance",with_distance)
        x,y = load_all_distance_metrics(test_size= 0.2, val_size= 0,with_distance= with_distance)
        x = x["test"]
        y = y["test"]
        x_l = list(x.values())

    else:
        if with_distance == True:
            print("2")
            x,y = load_everthing_with_distance(test_size= 0.2, val_size= 0)
            x = x["test"]
            y = y["test"]
        else:
            print("3")
            x, y = load_everthing(test_size= 0.2, val_size= 0)
            x = x["test"]
            y = y["test"]
        x_l = list(x["CosDist"].values())
    
    y_l = list(y.values())
    
    #We sample n different indexes

    if n != 0:
        sample_from = list(range(len(x_l)))
        index = random.sample(sample_from, n)
        #Here we get the actual values randomly sampled from the list
        x_final = [x_l[x] for x in index]
        labels = np.log10([y_l[x] for x in index])
    else:
        x_final = x_l
        labels = np.log10(y_l)
    
    pred = clf.predict(x_final)
    #labels = [item for sublist in labels for item in sublist]
    return pred, labels

def bootstrap(pipeline, param_grid,distance_metric,n = 100, with_dist = False, all_data = False):
    
    print("number of sample runs:", n)
    print("loading in data")
    result = []
    if with_dist:
        x,y = load_everthing_with_distance(test_size= 0.2, val_size=0.0)
        x = x["train"]
        y = y["train"]
    else:
        x,y = load_everthing(test_size= 0.2, val_size= 0)
        x = x["train"]
        y = y["train"]
    print("running bootstrap")
    for i in range(n):

        if i%10 == 0:
            print(i)

        ylist = list(y.values())
        xlist = list(x[distance_metric].values())
        index_to_pick_from = range(len(xlist))
        bts_index = [random.choice(index_to_pick_from) for _ in xlist]
        bts_x = [xlist[x] for x in bts_index]
        bts_y = np.log10([ylist[x][0] for x in bts_index])
        search = GridSearchCV(pipeline, param_grid, n_jobs=2,scoring= "r2")
        search.fit(bts_x, bts_y)
        result.append(search)
    return result

from utils.load import load_all_distance_metrics

import numpy as np
def bootstrap_all_distance_metrics(pipeline, param_grid,n = 100, with_dist = False):
    
    print("number of sample runs:", n)
    print("loading in data")
    result = []

    x,y = load_all_distance_metrics(test_size= 0.2, val_size=0.0, with_distance= with_dist)
    x = x["train"]
    y = y["train"]

        
    print("running bootstrap")
    for i in range(n):

        if i%10 == 0:
            print(i)
        xlist = list(x.values())
        ylist = list(y.values())
        index_to_pick_from = range(len(xlist))
        bts_index = [random.choice(index_to_pick_from) for _ in xlist]
        bts_x = [xlist[x] for x in bts_index]
        bts_y = np.log10([ylist[x][0] for x in bts_index])
        search = GridSearchCV(pipeline, param_grid, n_jobs=2,scoring= "r2")

        search.fit(bts_x, bts_y)
        result.append(search)
    return result
        

def bootstrap_continents(pipeline, param_grid,n = 100):
    print("running bootstrap")
    print("number of sample runs:", n)
    result = defaultdict(list)

    for i in range(n):
        if i%10 == 0:
            print(i)
        x_dict,y_dict = load_everthing_with_continents()
        x_dict = x_dict["CosDist"]
        for x,y in zip(x_dict.items(), y_dict.values()):
            
            continent = x[0]
            x = list(x[1].values())
            ylist = list(y.values())
            xlist = x
            index_to_pick_from = range(len(xlist))
            bts_index = [random.choice(index_to_pick_from) for _ in xlist]
            bts_x = [xlist[x] for x in bts_index]
            bts_y = np.log10([ylist[x][0] for x in bts_index])
            search = GridSearchCV(pipeline, param_grid, n_jobs=2,scoring= "r2")
            search.fit(bts_x, bts_y)
            result[continent].append(search)
    return result


def remove_outside_confidence_interval(n, li):
    """Removes values outside the n confidence interval"""
    li.sort()
    confidence_range = int(len(li) * (1 - n) / 2)
    lower_bound = li[confidence_range]
    upper_bound = li[-(confidence_range + 1)]
    return [x for x in li if x >= lower_bound and x <= upper_bound]





def gen_feature_dict_lasso(bootstrap_results, with_dist = False, all_distance = False):
    param_list = [x.best_estimator_.steps[-1][-1].coef_ for x in bootstrap_results]
    feature_list = []
    for file in os.listdir("data/fb_data/"):
        if all_distance:
            if (file.endswith(".csv")) and ("_" in file):
                #print(file)
                feature = file.replace(".csv","")
                feature_list.append(feature)
        else:
            if ("CosDist" in file) and (file.endswith(".csv")) and (file != "FBCosDist.csv"):
                feature = file.split("_")[1]
                feature = feature.replace(".csv","")
                feature_list.append(feature)
    if with_dist == True:
        feature_list.append("distance")
    d = defaultdict(list)
    for run in param_list:
        for feature, value in zip(feature_list,run):
            #print(feature)
            d[feature].append(value)

    #Removes values outside n% confidence interval
    for key, val in d.items():
        n = 0.95
        d[key] = remove_outside_confidence_interval(n,val)
    return d

def gen_feature_dict_rf(bootstrap_results, with_dist = False, all_distance = False):
    param_list = [x.best_estimator_.steps[-1][-1].feature_importances_ for x in bootstrap_results]
    feature_list = []
    for file in os.listdir("data/fb_data/"):
        if all_distance:
            if (file.endswith(".csv")) and ("_" in file):
                #print(file)
                feature = file.replace(".csv","")
                feature_list.append(feature)
        else:
            if ("CosDist" in file) and (file.endswith(".csv")) and (file != "FBCosDist.csv"):
                feature = file.split("_")[1]
                feature = feature.replace(".csv","")
                feature_list.append(feature)
    if with_dist == True:
        feature_list.append("distance")
    d = defaultdict(list)
    for run in param_list:
        for feature, value in zip(feature_list,run):
            #print(feature)
            d[feature].append(value)
    #Removes values outside n% confidence interval
    for key, val in d.items():
        n = 0.95
        d[key] = remove_outside_confidence_interval(n,val)
    return d


def gen_feature_dict(bootstrap_results, model_type, with_dist = False, all_distance = False):
    """NOT IN USE"""
    if model_type not in ["lasso","rf","ridge"]:
        raise ValueError('model_type not ["lasso","rf","ridge"]')
    if model_type in ["lasso", "ridge"]:
        param_list = [x.best_estimator_.steps[-1][-1].coef_ for x in bootstrap_results]
    elif model_type == "rf":
        param_list = [x.best_estimator_.steps[-1][-1].feature_importances_ for x in bootstrap_results]

    feature_list = []
    for file in os.listdir("data/fb_data/"):
        if all_distance:
            if (file.endswith(".csv")) and ("_" in file):
                #print(file)
                feature = file.replace(".csv","")
                feature_list.append(feature)
        else:
            if ("CosDist" in file) and (file.endswith(".csv")) and (file != "FBCosDist.csv"):
                feature = file.split("_")[1]
                feature = feature.replace(".csv","")
                feature_list.append(feature)
    if with_dist == True:
        feature_list.append("distance")
    d = defaultdict(list)
    for run in param_list:
        for feature, value in zip(feature_list,run):
            #print(feature)
            d[feature].append(value)

    #Removes values outside n% confidence interval
    for key, val in d.items():
        n = 0.95
        d[key] = remove_outside_confidence_interval(n,val)
    return d


def process(x,y):
    """This function converts the data into the train and test data. It also performs log transormation of labels(Y)"""
    if "CosDist" in x["train"]:
        x_train = dict()
        x_test = dict()
        for distance_metric, val in x["train"].items():
            x_train_temp = list(val.values())
            x_train[distance_metric] = x_train_temp
        for distance_metric, val in x["test"].items():
            x_test_temp = list(val.values())
            x_test[distance_metric] = x_test_temp
    else:
        x_train = x["train"]
        x_test = x["test"]
        x_train = list(x_train.values())
        x_test = list(x_test.values())
    y_train = y["train"]
    y_test = y["test"]
    y_train = [np.log10(x[0]) for x in list(y_train.values())]
    y_test = [np.log10(x[0]) for x in list(y_test.values())]
    return x_train, y_train, x_test, y_test

def file_name(with_distance, with_all_dist_metrics, pipeline, distance_metrics = None):
    
    name = ""

    if "Lasso_regressor" == pipeline.steps[-1][0]:
        name+= "LASSO "
    elif "rf" == pipeline.steps[-1][0]:
        name+= "RF "
    elif "Ridge_regressor" == pipeline.steps[-1][0]:
        name+= "RIDGE "
    if with_distance:
        name+= " with distance"
    if with_all_dist_metrics:
        name+= " all distance metrics"

    if "PCA" == pipeline.steps[-2][0]:
        name+= " with PCA" 
    
    if distance_metrics != None:
        name+= " " +distance_metrics
    return name


def run_confidence_interval(pipeline, param_grid,distance_metric, with_distance, all_distance_metrics, title):
    n = 1
    if n != 100:
        print(f"n={n} "*20)
    if "Lasso_regressor" == pipeline.steps[-1][0]:
        feature_dict_function = gen_feature_dict_lasso
    elif "rf" == pipeline.steps[-1][0]:
        feature_dict_function = gen_feature_dict_rf
    elif "Ridge_regressor" == pipeline.steps[-1][0]:
        feature_dict_function = gen_feature_dict_lasso

    
    if all_distance_metrics:
        s = bootstrap_all_distance_metrics(pipeline= pipeline, param_grid= param_grid,with_dist= with_distance, n = n)
        feature_dict = feature_dict_function(s, with_dist= with_distance, all_distance= all_distance_metrics)
        feature_dict = dict( sorted(feature_dict.items(), key=lambda x: x[0].lower()) )
    else:
        s = bootstrap(pipeline= pipeline, param_grid= param_grid,distance_metric=distance_metric,with_dist= with_distance, n = n)
        feature_dict = feature_dict_function(s, with_dist= with_distance)
    plot_confidence_interval(feature_dict,title)



