from utils.utilities import bootstrap, gen_feature_dict_d_trees
from utils.plots import plot_confidence_interval

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline



def run():
    pipe = Pipeline( [("StandardScaler",StandardScaler()), ('rf', RandomForestRegressor())]) 
    param_grid = { 'rf__max_features':[1.0], 'rf__n_estimators':[150], 'rf__max_depth':[None], 'rf__min_samples_split': [2], 'rf__min_samples_leaf': [2] }
    s = bootstrap(pipeline= pipe, param_grid= param_grid, n = 10)
    #t = [x.gridsearch for x in s]
    feature_dict = gen_feature_dict_d_trees(s)
    plot_confidence_interval(feature_dict,"Coefficient_estimate_d_tree")