import os
import pandas as pd
import joblib
import numpy as np
from utils.load import load_everthing
from sklearn.model_selection import GridSearchCV


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
        joblib.dump(self.gridsearch, self.model_name)
        d = pd.DataFrame(self.gridsearch.cv_results_)
        d = d[["params","mean_test_score"]]
        d["dataset"] =  [dataset for _ in range(len(d)) ]
        d["model"] =  [self.model_name for _ in range(len(d)) ]
        self.dataframe = d
        self.X = X
        self.Y = Y
        self.dataframe.to_csv(self.run_name, index = False)

def gridsearch(pipeline,param_grid, log_transform = True, update_merge_df = True):
    obj_list = []
    print("Loading in data")
    X_dict, Y_dict =load_everthing()


    for X_d, Y_d in zip(X_dict.items(),Y_dict.items()):
        dataset = X_d[0]
        X = list(X_d[1].values())
        Y = [x[0] for x in Y_d[1].values()]
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
    
    if update_merge_df == True:
        merge_dfs()
    return obj_list