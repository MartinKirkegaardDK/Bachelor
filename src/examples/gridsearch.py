from utils.load import load_everthing
#from sklearn.model_selection import train_test_split
from utils.utilities import result_object
import numpy as np
from sklearn.model_selection import GridSearchCV
from utils.utilities import gridsearch
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
#from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from utils.load import load_everthing_old
from sklearn.model_selection import RandomizedSearchCV



def model():
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid


    X_dict, Y_dict = load_everthing_old()

    X = list(X_dict.values())
    Y = [x[0] for x in Y_dict.values()]
    Y = np.log10(Y)



    pipe = Pipeline(
        [("normalize",Normalizer()),
        ('rf', RandomForestRegressor())])

    param_grid = {'rf__n_estimators': n_estimators}
             #   'rf__max_features': max_features,
              #  'rf__max_depth': max_depth,
               # 'rf__min_samples_split': min_samples_split,
               # 'rf__min_samples_leaf': min_samples_leaf,
               # 'rf__bootstrap': bootstrap}


    #rf_random = RandomizedSearchCV(pipe, param_distributions = param_grid)
    rf_random = gridsearch(pipe, param_grid)
    rf_random.fit(X, Y)
    print(rf_random.best_params_)





