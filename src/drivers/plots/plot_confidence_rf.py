from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from utils.utilities import bootstrap_continents
from utils.utilities import bootstrap
from utils.utilities import gen_feature_dict_rf
from utils.plots import plot_confidence_interval

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

n = 100

def confidence_plot_rf_continents():
    pipe = Pipeline( [("StandardScaler",StandardScaler()), ('rf', RandomForestRegressor())]) 
    param_grid = {}
    s = bootstrap_continents(pipeline= pipe, param_grid= param_grid, n = n)
    for continent, value in s.items():
        feature_dict = gen_feature_dict_rf(value)
        plot_confidence_interval(feature_dict,f"Coefficient_estimate_d_tree_{continent}",continent= continent)

def confidence_plot_rf_without_dist():
    pipe = Pipeline( [("StandardScaler",StandardScaler()), ('rf', RandomForestRegressor())]) 
    param_grid = {}
    s = bootstrap(pipeline= pipe, param_grid= param_grid, n = n)
    feature_dict = gen_feature_dict_rf(s)
    plot_confidence_interval(feature_dict,"Coefficient_estimate_d_tree")

def confidence_plot_rf_with_dist():
    pipe = Pipeline( [("StandardScaler",StandardScaler()), ('rf', RandomForestRegressor())]) 
    param_grid = {}
    s = bootstrap(pipeline= pipe, param_grid= param_grid, n = n,with_dist= True)
    feature_dict = gen_feature_dict_rf(s,with_dist= True)
    plot_confidence_interval(feature_dict,"Coefficient_estimate_d_tree_with_distance")

def run():
    print("plotting confidence_plot_rf_without_dist")
    confidence_plot_rf_without_dist()
    print("plotting confidence_plot_rf_with_dist")
    confidence_plot_rf_with_dist()