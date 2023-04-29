
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from utils.utilities import gridsearch_all_distance_metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA


def lasso_without_dist_all_distance():
    pipe =  Pipeline([("StandardScaler",StandardScaler()),("PCA",PCA()),("Lasso_regressor",LassoCV(max_iter= 100000, tol= 0.001))])
    param_grid = {"PCA__n_components":list(range(0,40,2))}
    gridsearch_all_distance_metrics(pipe, param_grid, with_distance= False)

def lasso_with_dist_all_distance():
    pipe =  Pipeline([("StandardScaler",StandardScaler()),("PCA",PCA()),("Lasso_regressor",LassoCV(max_iter= 100000, tol= 0.001))])
    param_grid = {"PCA__n_components":list(range(0,40,2))}
    gridsearch_all_distance_metrics(pipe, param_grid, with_distance= True)

def rf_without_dist_all_distance():
    pipe = Pipeline( [("StandardScaler",StandardScaler()),("PCA",PCA()), ('rf', RandomForestRegressor())]) 
    param_grid = {"PCA__n_components":list(range(0,40,2))}
    gridsearch_all_distance_metrics(pipe, param_grid, with_distance= False)

def rf_with_dist_all_distance():
    pipe = Pipeline( [("StandardScaler",StandardScaler()),("PCA",PCA()), ('rf', RandomForestRegressor())]) 
    param_grid = {"PCA__n_components":list(range(0,40,2))}
    gridsearch_all_distance_metrics(pipe, param_grid, with_distance= True)

def run():
    print("running lasso_without_dist_all_distance")
    lasso_without_dist_all_distance()
    print("running lasso_with_dist_all_distance")
    lasso_with_dist_all_distance()
    print("running rf_without_dist_all_distance")
    rf_without_dist_all_distance()
    print("running rf_with_dist_all_distance")
    rf_with_dist_all_distance()

