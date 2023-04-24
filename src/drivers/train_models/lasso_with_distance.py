
from utils.utilities import gridsearch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV


def run():
    pipe =  Pipeline([("StandardScaler",StandardScaler()),("Lasso_regressor",LassoCV(max_iter= 100000, tol= 0.001))])
    param_grid = {}
    gridsearch(pipeline= pipe, param_grid= param_grid,with_distance= True)

