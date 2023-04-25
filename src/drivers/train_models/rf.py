#from utils.utilities import gridsearch_continent
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from utils.utilities import gridsearch

#Skal der også inkluderes pca?

def rf_with_dist():
    pipe = Pipeline( [("StandardScaler",StandardScaler()), ('rf', RandomForestRegressor())]) 
    param_grid = { 'rf__max_features':[1.0], 'rf__n_estimators':[150], 'rf__max_depth':[None], 'rf__min_samples_split': [2], 'rf__min_samples_leaf': [2] }
    gridsearch(pipe, param_grid, with_distance= True)

def rf_without_dist():
    pipe = Pipeline( [("StandardScaler",StandardScaler()), ('rf', RandomForestRegressor())]) 
    param_grid = { 'rf__max_features':[1.0], 'rf__n_estimators':[150], 'rf__max_depth':[None], 'rf__min_samples_split': [2], 'rf__min_samples_leaf': [2] }
    gridsearch(pipe,param_grid,with_distance= False)

def run():
    print("training rf_with_dist")
    rf_with_dist()
    print("training rf_without_dist")
    rf_without_dist()

