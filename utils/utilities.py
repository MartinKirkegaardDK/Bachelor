import os
import pandas as pd
import joblib
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
    def __init__(self, grid_search: object, data_type: str,X,Y):
        self.gridsearch = grid_search
        self.data_type = data_type
        self.model_name = naming("models/","test","pkl")
        self.run_name = naming("runs/","test","csv")
        joblib.dump(self.gridsearch, self.model_name)
        d = pd.DataFrame(self.gridsearch.cv_results_)
        d = d[["params","mean_test_score"]]
        d["dataset"] =  [data_type for _ in range(len(d)) ]
        d["model"] =  [self.model_name for _ in range(len(d)) ]
        self.dataframe = d
        self.X = X
        self.Y = Y
        self.dataframe.to_csv(self.run_name, index = False)
    