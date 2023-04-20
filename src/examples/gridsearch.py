from utils.utilities import gridsearch
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

pipe = Pipeline(
    [("normalize",StandardScaler()),
    ('SVM', SVR())])

param_grid = {
            "SVM__kernel": ["poly","linear"]
        }


gridsearch(pipe, param_grid)