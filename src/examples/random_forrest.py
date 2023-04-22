from utils.load import load_everthing
#from sklearn.model_selection import train_test_split
from utils.utilities import result_object
import numpy as np
from sklearn.model_selection import GridSearchCV
from utils.utilities import gridsearch, gridsearchJulie
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
#from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from utils.load import load_everthing_old
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.model_selection import train_test_split

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler





def model():
    # Number of trees in random forest
    n_estimators = [160,180,200,220,240,260,280,300] #[180,190,200,210,220,230]
    max_features = ['auto', 'sqrt']
    min_samples_split = [2,3,4,5,6]
    min_samples_leaf = [2,3,4,5,6]
    bootstrap = [True, False]


    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 500, num = 10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    min_samples_split = [2, 3,4,5,6, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4,6,8]
    # Method of selecting samples for training each tree
    bootstrap = ['memory', 'steps', 'verbose']
    # Create the random grid
    param_grid = {'rf__n_estimators': n_estimators,
                'rf__max_features': max_features,
                'rf__max_depth': max_depth,
                'rf__min_samples_split': min_samples_split,
                'rf__min_samples_leaf': min_samples_leaf}


   # param_grid = {
    #            'rf__n_estimators':n_estimators, 'rf__max_depth':[None], 'rf__min_samples_split': min_samples_split, 'rf__min_samples_leaf': min_samples_leaf
     #       }

    pipe = Pipeline(
        [("StandardScaler",StandardScaler()),
        ('rf', RandomForestRegressor())])


    rf_random = gridsearchJulie(pipe, param_grid)
    








