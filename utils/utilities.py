import os
import pandas as pd
import joblib
import numpy as np
import random
from utils.load import load_everthing, get_feature_names, load_everthing_old
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression
from utils.load import load
from collections import defaultdict
from utils.load import load_everthing_with_countries

def naming(path: str, name: str, extension: str) -> str:
    return path + name + "_" + str(len(os.listdir(path)) + 1) +"." + extension

def merge_dfs():
    path = "runs/"
    df_list = []
    for elm in os.listdir(path):
        if "merged" in elm:
            continue
        df_list.append(pd.read_csv(path + elm))
    df = pd.concat(df_list)
    df.to_csv("runs/merged_df.csv",index = False)

class result_object():
    def __init__(self, grid_search: object, dataset: str,X,Y,region):
        self.gridsearch = grid_search
        self.data_type = dataset
        self.model_name = naming("models/","test","pkl")
        self.run_name = naming("runs/","test","csv")
        self.pipeline = grid_search.estimator
        joblib.dump(self.gridsearch, self.model_name) 
        d = pd.DataFrame(self.gridsearch.cv_results_)
        d = d[["params","mean_test_score"]]
        d["dataset"] =  [dataset for _ in range(len(d)) ]
        #d["score"] = grid_search.best_score_
        d["model_file_name"] =  [self.model_name for _ in range(len(d)) ]
        d["pipeline"] = self.pipeline
        d["region"] = self.region = region
        d["model_name"] = [self.pipeline[-1] for _ in range(len(d)) ] 
        self.dataframe = d
        self.X = X
        self.Y = Y
        
        
        self.dataframe.to_csv(self.run_name, index = False)

def gridsearch():#(pipeline,param_grid, log_transform = True, update_merge_df = True):
    obj_list = []
    print("Loading in data")
    X_dict, Y_dict = load_everthing()


    for X_d, Y_d in zip(X_dict.items(),Y_dict):
        dataset = X_d[0]
        X = list(X_d[1].values())
        Y = [x[0] for x in Y_dict.values()]
        if log_transform == True:
            Y = np.log10(Y)
        print(dataset)

        print("running gridsearch")
        search = GridSearchCV(pipeline, param_grid, n_jobs=2,scoring= "r2")

        search.fit(X, Y)
        print("Best parameter (CV score=%0.3f):" % search.best_score_)
        print(search.best_params_)
        obj = result_object(search, dataset,X,Y,"world")
        obj_list.append(obj)
        print("-"*75)
    if update_merge_df == True:
        merge_dfs()
    return obj_list



def gridsearch_continent(pipeline,param_grid, log_transform = True, update_merge_df = True):
    obj_list = []
    print("Loading in data")
    x_dict, y_dict = load_everthing_with_countries()
    for distance_metrics in x_dict.keys():
        print(distance_metrics)
        for x_d, labels in zip(x_dict[distance_metrics].items(),y_dict.values()):
            continent = x_d[0]
            X = list(x_d[1].values())
            labels = list(labels.values())
            labels = [x[0] for x in labels]
            print(continent)
            if log_transform == True:
                Y = np.log10(labels)
           
            print("running gridsearch")
            search = GridSearchCV(pipeline, param_grid, n_jobs=2,scoring= "r2")

            search.fit(X, Y)
            print("Best parameter (CV score=%0.3f):" % search.best_score_)
            print(search.best_params_)
            obj = result_object(search, distance_metrics,X,Y,country)
            obj_list.append(obj)
            print("-"*75)
        if update_merge_df == True:
            merge_dfs()
    return obj_list

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

def gridsearchJulie(pipeline, param_grid):
    """

    print("Loading in data")
    X_dict, Y_dict =load_everthing()

    for X_d, Y_d in zip(X_dict.items(),Y_dict):
        dataset = X_d[0]

        names = get_feature_names(dataset)
        
        X = list(X_d[1].values())
        Y = [x[0] for x in Y_dict.values()]
        Y = np.log10(Y)
        print(dataset) # The metric we are looking at
        print("running gridsearch")

        search = GridSearchCV(pipeline, param_grid, n_jobs=2,scoring= "r2")
        search.fit(X, Y)
        print("Best Parameters", search.best_params_)
        print("Best parameter (CV score=%0.3f):" % search.best_score_)
        print("-"*75)

    """
    # All data together
    X_dict, Y_dict = load_everthing_old()
    distance  = pd.read_csv("data/distance_data/processed_distances.csv")
    distance = distance['0'].to_list()

    for key, value, new_number in zip(X_dict.keys(), X_dict.values(), distance):
        X_dict[key] = value + (new_number,)


    X = list(X_dict.values())
    Y = [i[0] for i in Y_dict.values()]
    Y = np.log10(Y)

    print("all metrics together without distance") # The metric we are looking at
    print("running gridsearch...")
    search = GridSearchCV(pipeline, param_grid, n_jobs=2,scoring= "r2")
    search.fit(X, Y)
    print("Best Parameters", search.best_params_)
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print("-"*75)


    for dist, X_d, Y_d in zip(distance, X_dict.values(),Y_dict.values()):
        X = list(X_d)
        X.append(dist) #adding distance
        Y = np.log10(Y_d[0]).to_list()
        print("all metrics together with distance") # The metric we are looking at
        print("running gridsearch...")
        search = GridSearchCV(pipeline, param_grid, n_jobs=2,scoring= "r2")
        search.fit(X, Y)
        print("Best Parameters", search.best_params_)
        print("Best parameter (CV score=%0.3f):" % search.best_score_)
        print("-"*75)

    return 


def get_pred_and_labels(clf,n = 50 ):
    """samples n amount of datapoints and uses the model clf to predict
    if n = 0, then we dont sample anything and use the entire dataset"""
    x, y = load_everthing()
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

    pred = clf.best_estimator_.predict(x_final)
    labels = [item for sublist in labels for item in sublist]
    return pred, labels


def bootstrap(pipeline, param_grid,n = 100):
    print("running bootstrap")
    print("number of sample runs:", n)
    result = []
    for i in range(n):
        if i%10 == 0:
            print(i)
        x,y = load_everthing()
        ylist = list(y.values())
        xlist = list(x["CosDist"].values())
        index_to_pick_from = range(len(xlist))
        bts_index = [random.choice(index_to_pick_from) for _ in xlist]
        bts_x = [xlist[x] for x in bts_index]
        bts_y = np.log10([ylist[x][0] for x in bts_index])
        
        param_grid = {}
        search = GridSearchCV(pipeline, param_grid, n_jobs=2,scoring= "r2")
        search.fit(bts_x, bts_y)
        result.append(search)
    return result

def remove_outside_confidence_interval(n, li):
    """Removes values outside the n confidence interval"""
    li.sort()
    confidence_range = int(len(li) * (1 - n) / 2)
    lower_bound = li[confidence_range]
    upper_bound = li[-(confidence_range + 1)]
    return [x for x in li if x >= lower_bound and x <= upper_bound]



def gen_feature_dict(bootstrap_results):
    param_list = [x.best_estimator_.steps[-1][-1].coef_ for x in bootstrap_results]
    feature_list = []
    for file in os.listdir("data/fb_data/"):
        if ("CosDist" in file) and (file.endswith(".csv")) and (file != "FBCosDist.csv"):
            feature = file.split("_")[1]
            feature = feature.replace(".csv","")
            feature_list.append(feature)
    d = defaultdict(list)
    for run in param_list:
        for feature, value in zip(feature_list,run):
            d[feature].append(value)

    #Removes values outside n% confidence interval
    for key, val in d.items():
        n = 0.95
        d[key] = remove_outside_confidence_interval(n,val)
    return d