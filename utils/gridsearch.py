
from utils.load import load_all_distance_metrics, load_everthing_with_distance, load_everthing, load_everthing_with_continents
from utils.utilities import result_object, merge_dfs
import numpy as np
from sklearn.model_selection import GridSearchCV

def gridsearch_old(pipeline,param_grid, remove_threshold = 0,log_transform = True, update_merge_df = True, with_distance = False):
    """If the remove_threshold is set to a positive value above 0, then it removes the labels
    which has a score of lower then thre remove_threshold"""
    obj_list = []
    print("Loading in data")
    if with_distance == True:
        X_dict, Y_dict = load_everthing_with_distance(test_size = 0.2, val_size= 0)
    else:
        X_dict, Y_dict =load_everthing(test_size= 0.2,val_size=0)
    X_dict = X_dict["train"]
    Y_dict = Y_dict["train"]
    if remove_threshold != 0:
        #This was removed?
        #X_dict, Y_dict, _ = remove_under_threshold(remove_threshold, X_dict,Y_dict)
        pass
    

    for X_d, Y_d in zip(X_dict.items(),Y_dict):
        dataset = X_d[0]
        print(dataset)
        #if (dataset in ["CosDist","EucDist","HetDist"]) and (with_distance == True):
        #    continue
        X = list(X_d[1].values())
        Y = [x[0] for x in Y_dict.values()]
        if log_transform == True:
            Y = np.log10(Y)
        #print(dataset)

        print("running gridsearch")
        search = GridSearchCV(pipeline, param_grid, n_jobs=2,scoring= "r2")

        search.fit(X, Y)
        print("Best parameter (CV score=%0.3f):" % search.best_score_)
        print(search.best_params_)
        obj = result_object(search, dataset,X,Y,"world",with_distance)
        obj_list.append(obj)
        print("-"*75)
    if update_merge_df == True:
        merge_dfs()
    return obj_list




def gridsearch_all_distance_metrics(pipeline,param_grid, remove_threshold = 0,log_transform = True, update_merge_df = True, with_distance = False):
    import numpy as np
    if with_distance == True:
        x,y = load_all_distance_metrics(test_size = 0.2, val_size= 0, with_distance = True)
        x = x["train"]
        y = y["train"]
    else:
        x,y = load_all_distance_metrics(test_size = 0.2, val_size= 0, with_distance = False)
        x = x["train"]
        y = y["train"]
    
    X = list(x.values())
    Y = [x[0] for x in y.values()]
    print(Y)
    if log_transform == True:
        Y = np.log10(Y)
    #print(dataset)
    dataset = "all_distance_metrics"
    print("running gridsearch")
    search = GridSearchCV(pipeline, param_grid, n_jobs=2,scoring= "r2")

    search.fit(X, Y)
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)
    obj = result_object(search, dataset,X,Y,"world",with_distance)
    return obj


def gridsearch_continent(pipeline,param_grid, log_transform = True, update_merge_df = True):
    obj_list = []
    print("Loading in data")
    x_dict, y_dict = load_everthing_with_continents()
    x_dict = x_dict["train"]
    y_dict = y_dict["train"]
    for distance_metrics in x_dict.keys():
        print(distance_metrics)
        for x_d, y_d in zip(x_dict[distance_metrics].items(),y_dict.items()):
            continent = x_d[0]
            X = list(x_d[1].values())
            labels = list(y_d[1].values())
            labels = [x[0] for x in labels]

            if log_transform == True:
                Y = np.log10(labels)
           
            print("running gridsearch")
            search = GridSearchCV(pipeline, param_grid, n_jobs=2,scoring= "r2")

            search.fit(X, Y)
            print("Best parameter (CV score=%0.3f):" % search.best_score_)
            print(search.best_params_)
            obj = result_object(search, distance_metrics,X,Y,continent,False)
            obj_list.append(obj)
            print("-"*75)
        if update_merge_df == True:
            merge_dfs()
    return obj_list
