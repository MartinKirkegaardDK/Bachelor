from utils.load import load_everthing
from sklearn.model_selection import train_test_split
from utils.utilities import result_object

#Preprocessing tools
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline

#Classifiers
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR



def run():
    obj_list = []
    print("Loading in data")
    X_dict, Y_dict =load_everthing()


    for X_d, Y_d in zip(X_dict.items(),Y_dict.items()):
        dataset = X_d[0]
        X = list(X_d[1].values())
        Y = [x[0] for x in Y_d[1].values()]


        pipe = Pipeline(
            [("normalize",Normalizer()),
            ('SVM', SVR())])
        #X_train, X_test, Y_train, Y_test = train_test_split(
        #X, Y, test_size=0.15, random_state=42)

        #reg = LinearRegression().fit(X_train, Y_train)
        #pipe.fit(X_train, Y_train)
        #score = pipe.score(X_test, Y_test)
        
        #param_grid = {
        #    "SVM__kernel": ["rbf","linear","poly"]
        #}
        param_grid = {
            "SVM__kernel": ["poly","linear"]
        }
        print("running gridsearch")
        search = GridSearchCV(pipe, param_grid, n_jobs=2,scoring= "r2")
        



        search.fit(X, Y)
        print("Best parameter (CV score=%0.3f):" % search.best_score_)
        print(search.best_params_)
        obj = result_object(search, dataset,X,Y)
        obj_list.append(obj)
    return obj_list



