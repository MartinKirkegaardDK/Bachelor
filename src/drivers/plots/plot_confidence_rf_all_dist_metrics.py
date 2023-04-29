from utils.utilities import bootstrap, gen_feature_dict_rf
from utils.plots import plot_confidence_interval
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from utils.utilities import bootstrap_all_distance_metrics

n = 100

def confidence_plot_rf_without_dist_all_metrics():
    pipe = Pipeline( [("StandardScaler",StandardScaler()), ('rf', RandomForestRegressor())]) 
    param_grid = {}
    s = bootstrap_all_distance_metrics(pipe, param_grid, n = 10)
    feature_dict = gen_feature_dict_rf(s)
    feature_dict = dict( sorted(feature_dict.items(), key=lambda x: x[0].lower()) )
    plot_confidence_interval(feature_dict,"Coefficient_estimate_for_RF_without_distance_all_metrics")

def confidence_plot_rf_with_dist_all_metrics():
    pipe = Pipeline( [("StandardScaler",StandardScaler()), ('rf', RandomForestRegressor())]) 
    s = bootstrap_all_distance_metrics(pipeline= pipe, param_grid= {},with_dist= True, n = 1)
    feature_dict = gen_feature_dict_rf(s, with_dist= True)
    feature_dict = dict( sorted(feature_dict.items(), key=lambda x: x[0].lower()) )
    plot_confidence_interval(feature_dict,"Coefficient_estimate_for_RF_with_distance_all_metrics")

def run():
    print("plotting confidence_plot_rf_without_dist_all_metrics")
    confidence_plot_rf_without_dist_all_metrics()
    print("plotting confidence_plot_rf_with_dist_all_metrics")
    confidence_plot_rf_with_dist_all_metrics()