#from utils.utilities import gridsearch_continent
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from utils.utilities import gridsearch
from utils.utilities import gridsearch_continent
from sklearn.linear_model import LassoCV



def continents_lasso():
    pipe = Pipeline([("StandardScaler",StandardScaler()),("Lasso_regressor",LassoCV(max_iter= 100000, tol= 0.001))])
    param = {}
    gridsearch_continent(pipe, param)

def continents_rf():
    pipe = Pipeline( [("StandardScaler",StandardScaler()), ('rf', RandomForestRegressor())]) 
    param_grid = { 'rf__max_features':[1.0], 'rf__n_estimators':[150], 'rf__max_depth':[None], 'rf__min_samples_split': [2], 'rf__min_samples_leaf': [2] }
    gridsearch(pipeline= pipe, param_grid= param_grid)

def run():
    print("training continents_rf")
    continents_rf()
    print("training continents_lasso")
    continents_lasso()

