
from utils.utilities import gridsearch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV


def lasso_without_dist():
    pipe =  Pipeline([("StandardScaler",StandardScaler()),("Lasso_regressor",LassoCV(max_iter= 100000, tol= 0.001))])
    param_grid = {}
    gridsearch(pipeline= pipe, param_grid= param_grid,with_distance= False)

def lasso_with_dist():
    pipe =  Pipeline([("StandardScaler",StandardScaler()),("Lasso_regressor",LassoCV(max_iter= 100000, tol= 0.001))])
    param_grid = {}
    gridsearch(pipeline= pipe, param_grid= param_grid,with_distance= True)

def run():
    print("training lasso with dist")
    lasso_with_dist()
    print("training lasso without dist")
    lasso_without_dist()

