from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from utils.utilities import gridsearch
from sklearn.pipeline import Pipeline


def train_models(dict):
    """Train multiple models from single dict"""
    for key, val in dict.items():
        gridsearch(Pipeline(val["pipeline"]),val["param_grid"])

def lasso():
    """This is the linear regression model using lasso"""
    model_dict = dict()
    model_dict["Lasso_regressor"]["param_grid"] = {
    }
    model_dict["Lasso_regressor"]["pipeline"] =   [("StandardScaler",StandardScaler()),("Lasso_regressor",LassoCV(max_iter= 10000))]
    train_models(model_dict)
    
