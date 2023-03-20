from utils.load import load_everthing
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import numpy as np

def run():
    X_dict, Y_dict =load_everthing()

    X = list(X_dict.values())
    Y = [x[0] for x in Y_dict.values()]

    Y = np.log10(Y)
    
    reg = RandomForestRegressor(max_depth=2, random_state=0)

    reg.fit(X, Y)
    print(reg.score(X, Y))
