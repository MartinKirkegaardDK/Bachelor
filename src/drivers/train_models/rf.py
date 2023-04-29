#from utils.utilities import gridsearch_continent
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from utils.utilities import gridsearch
import numpy as np

#Skal der også inkluderes pca?

def rf():
            # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 200, num = 7)]
    #max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(start = 50, stop = 100, num = 7)]
    max_depth.append(None)
    min_samples_split = [1,2, 3]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 3]
    # Method of selecting samples for training each tree
    # Create the random grid
    param_grid = {
                "PCA__n_components": [None,2,3,4],
                'rf__n_estimators': n_estimators,
                'rf__max_depth': max_depth,
                'rf__min_samples_split': min_samples_split,
                'rf__min_samples_leaf': min_samples_leaf}


    # param_grid = {
    #            'rf__n_estimators':n_estimators, 'rf__max_depth':[None], 'rf__min_samples_split': min_samples_split, 'rf__min_samples_leaf': min_samples_leaf
    #       }

    pipe = Pipeline(
        [("StandardScaler",StandardScaler()),
         ("PCA",PCA()),
        ('rf', RandomForestRegressor())])
    return pipe, param_grid

def rf_with_dist():


    #pipe, param_grid = rf()
    pipe = Pipeline( [("StandardScaler",StandardScaler()), ('rf', RandomForestRegressor())]) 
    param_grid = {}
    gridsearch(pipe, param_grid, with_distance= True)

def rf_without_dist():
    #pipe, param_grid = rf()
    pipe = Pipeline( [("StandardScaler",StandardScaler()), ('rf', RandomForestRegressor())]) 
    param_grid = {}
    gridsearch(pipe,param_grid,with_distance= False)

def run():
    print("training rf_with_dist")
    rf_with_dist()
    print("training rf_without_dist")
    rf_without_dist()

