from utils.load import load_everthing
from sklearn.model_selection import train_test_split

#Preprocessing tools
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#Classifiers
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

def run():
    X_dict, Y_dict =load_everthing()

    X = list(X_dict.values())
    Y = [x[0] for x in Y_dict.values()]

    # pipe = Pipeline(
    #     [('scaler', StandardScaler()), 
    #      ('Linear regression', LinearRegression())])

    pipe = Pipeline(
        [('scaler', StandardScaler()), 
         ('SVM', SVR())])
    X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.15, random_state=42)

    reg = LinearRegression().fit(X_train, Y_train)
    pipe.fit(X_train, Y_train)
    score = pipe.score(X_test, Y_test)
    param_grid = {
        "SVM__kernel": ["rbf","linear","poly"]
    }
    print("running gridsearch")
    search = GridSearchCV(pipe, param_grid, n_jobs=2)
    search.fit(X, Y)
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)