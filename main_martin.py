from src.examples.gridsearch import run
#t = run()
from utils.utilities import merge_dfs



from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

from utils.utilities import gridsearch


pipe = Pipeline(
    [("normalize",Normalizer()),
    ('SVM', SVR())])

param_grid = {
            "SVM__kernel": ["poly","linear"]
        }

gridsearch()



merge_dfs()
