from utils.utilities import gridsearch_continent
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV



def run():
    model_dict = dict()
    model_dict["param_grid"] = {
    }
    model_dict["pipeline"] =   Pipeline([("StandardScaler",StandardScaler()),("Lasso_regressor",LassoCV(max_iter= 100000, tol= 0.001))])
    gridsearch_continent(model_dict["pipeline"],model_dict["param_grid"])