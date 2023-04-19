from utils.load import load_everthing
#from sklearn.model_selection import train_test_split
from utils.utilities import result_object
import numpy as np
from sklearn.model_selection import GridSearchCV
from utils.utilities import gridsearchJulie
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
   # n_estimators = [int(x) for x in np.linspace(start = 2, stop = 40, num = 10)]
    n_estimators = [180,200,220] #[180,190,200,210,220,230]
    #n_estimators.append(100)
    #n_estimators.append(200)

    # Number of features to consider at every split
    #max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = []#[int(x) for x in np.linspace(2,20, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2,3,4]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [2,3,4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid


    X_dict, Y_dict = load_everthing_old()

    X = list(X_dict.values())
    #X = preprocessing.normalize(X)

    y = [x[0] for x in Y_dict.values()]
    y = np.log10(y)

    print(X)
    pipe = Pipeline(
        [("StandardScaler",StandardScaler()),
        ('rf', RandomForestRegressor())])


    param_grid = {
                'rf_max_features':1.0, 'rf__n_estimators':150, 'rf__max_depth':None, 'rf__min_samples_split': 2, 'rf__min_samples_leaf': 2
            }


    rf_random = gridsearchJulie(pipe, param_grid)




       # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
   # regr = RandomForestRegressor(max_depth=2, random_state=0)
   # regr.fit(X_train, y_train)

   # y_pred = regr.predict(X_test)


   # print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
   # print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
   # print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))







