from utils.ml_lib import train_models
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from utils.utilities import gridsearch


def run():
    model_dict = dict()
    model_dict = {
        "Linear_regressor": dict(),
        "Ridge_regressor": dict(),
        "Lasso_regressor": dict(),
        "ElasticNet_regressor": dict()
    }

    model_dict["Linear_regressor"]["param_grid"] = {}
    model_dict["Linear_regressor"]["pipeline"] =   [("StandardScaler",StandardScaler()),("Linear_regressor",LinearRegression())]


    model_dict["Ridge_regressor"]["param_grid"] = {
        "Ridge_regressor__alpha": [1,10,25,100]
    }
    model_dict["Ridge_regressor"]["pipeline"] =   [("StandardScaler",StandardScaler()),("Ridge_regressor",Ridge(max_iter=10000))]



    model_dict["Lasso_regressor"]["param_grid"] = {
    }
    model_dict["Lasso_regressor"]["pipeline"] =   [("StandardScaler",StandardScaler()),("Lasso_regressor",LassoCV(max_iter= 10000))]


    model_dict["ElasticNet_regressor"]["param_grid"] = {
        "ElasticNet_regressor__alpha": [1,10,25,100]
    }
    model_dict["ElasticNet_regressor"]["pipeline"] =  [("StandardScaler",StandardScaler()),("ElasticNet_regressor",ElasticNet(max_iter=10000))]

    train_models(model_dict)