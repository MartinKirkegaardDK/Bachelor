from utils.utilities import gridsearch
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor


def run():
    pipe = Pipeline(
        [("StandardScaler",StandardScaler()),
        ('rf', RandomForestRegressor())])


    param_grid = {
                'rf__max_features':[1.0], 'rf__n_estimators':[150], 'rf__max_depth':[None], 'rf__min_samples_split': [2], 'rf__min_samples_leaf': [2]
            }

    gridsearch(pipe, param_grid = param_grid,with_distance= False)