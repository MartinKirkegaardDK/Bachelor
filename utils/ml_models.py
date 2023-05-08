from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor


def lasso():
    pipe =  Pipeline([("StandardScaler",StandardScaler()),("Lasso_regressor",LassoCV(max_iter= 100000, tol= 0.001))])
    param_grid = {"Lasso_regressor__alphas":[1,2,3]}
    return pipe, param_grid

def lasso_pca():
    pipe =  Pipeline([("StandardScaler",StandardScaler()),("PCA",PCA()),("Lasso_regressor",LassoCV(max_iter= 100000, tol= 0.001))])
    param_grid = {"PCA__n_components":list(range(0,40,2))}
    return pipe, param_grid

def rf():
    pipe = Pipeline( [("StandardScaler",StandardScaler()), ('rf', RandomForestRegressor())]) 
    param_grid = {"rf__criterion": ["squared_error"],
                  #"rf__criterion": ["squared_error","absolute_error"],
                  "rf__max_depth": [None, 3,5,7]}
    return pipe, param_grid


def rf_pca():
    pipe = Pipeline( [("StandardScaler",StandardScaler()),("PCA",PCA()), ('rf', RandomForestRegressor())]) 
    param_grid = {"PCA__n_components":list(range(0,40,2)),
                  "rf__criterion": ["squared_error"],
                  #"rf__criterion": ["squared_error","absolute_error"],
                  "rf__max_depth": [None, 3,5,7]}
    return pipe, param_grid

def ridge():
    pipe =  Pipeline([("StandardScaler",StandardScaler()),("Ridge_regressor",RidgeCV())])
    param_grid = {}
    return pipe, param_grid

