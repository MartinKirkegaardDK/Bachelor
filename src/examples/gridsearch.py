from utils.load import load_everthing
#from sklearn.model_selection import train_test_split
from utils.utilities import result_object
import numpy as np
from sklearn.model_selection import GridSearchCV
from utils.utilities import gridsearch
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
#from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier


pipe = Pipeline(
    [("normalize",Normalizer()),
    ('SVM', SVR())])

param_grid = {
            "SVM__kernel": ["poly","linear"]
        }

gridsearch(pipe, param_grid)