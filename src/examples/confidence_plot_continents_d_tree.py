import pandas as pd
from utils.utilities import gridsearch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from utils.utilities import bootstrap_continents
from utils.utilities import gen_feature_dict_d_trees
from utils.plots import plot_confidence_interval


def run():
    pipe = Pipeline( [("StandardScaler",StandardScaler()), ('rf', RandomForestRegressor())]) 
    param_grid = { 'rf__max_features':[1.0], 'rf__n_estimators':[150], 'rf__max_depth':[None], 'rf__min_samples_split': [2], 'rf__min_samples_leaf': [2] }
    s = bootstrap_continents(pipeline= pipe, param_grid= param_grid, n = 100)
    for continent, value in s.items():
        feature_dict = gen_feature_dict_d_trees(value)
        plot_confidence_interval(feature_dict,f"Coefficient_estimate_d_tree_{continent}",continent= continent)