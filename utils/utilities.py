import os
import pandas as pd
import joblib
import numpy as np
from utils.load import load_everthing, get_feature_names
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression


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
    def __init__(self, grid_search: object, dataset: str,X,Y):
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
        d["model_name"] = [self.pipeline[-1] for _ in range(len(d)) ] 
        self.dataframe = d
        self.X = X
        self.Y = Y
        
        self.dataframe.to_csv(self.run_name, index = False)

def gridsearch(pipeline,param_grid, log_transform = True, update_merge_df = True):
    obj_list = []
    print("Loading in data")
    X_dict, Y_dict =load_everthing()


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
        obj = result_object(search, dataset,X,Y)
        obj_list.append(obj)
        print("-"*75)
    if update_merge_df == True:
        merge_dfs()
    return obj_list





def gridsearchJulie(pipeline,param_grid, log_transform = True, update_merge_df = True):
    obj_list = []
    print("Loading in data")
    X_dict, Y_dict =load_everthing()

    for X_d, Y_d in zip(X_dict.items(),Y_dict):
        dataset = X_d[0]

        names = get_feature_names(dataset)
        
        X = list(X_d[1].values())
        Y = [x[0] for x in Y_dict.values()]
        if log_transform == True:
            Y = np.log10(Y)
        print(dataset) # The metric we are looking at
        print("running gridsearch")
        search = GridSearchCV(pipeline, param_grid, n_jobs=2,scoring= "r2")
       
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
        search.fit(X_train, y_train)

        y_pred = search.predict(X_test)
        print('\n\n')
        print("Best Parameters", search.best_params_)
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
            
        select = SelectKBest(score_func=f_regression, k=1)
        
        z = select.fit_transform(X, Y) 
        mask = select.get_support()
 

        new_features = [] # The list of your K best features

        for bool_val, feature in zip(mask, names):
            if bool_val:
                new_features.append(feature)
        print(new_features)
       # print("Best parameter (CV score=%0.3f):" % search.best_score_)
       # print(search.best_params_)
        obj = result_object(search, dataset,X,Y)
        obj_list.append(obj)
        print("-"*75)

   # if update_merge_df == True:
    #    merge_dfs()
   
    return obj_list