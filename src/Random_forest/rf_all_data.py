
from utils.load import load_everthing_old

from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor 

import matplotlib.pyplot as plt
import pandas as pd



def rf_with_distance():

    X_dict, Y_dict = load_everthing_old()
    distance  = pd.read_csv("data/distance_data/processed_distances.csv")
    distance["0"] = distance["0"].fillna(0)
    distance = distance['0'].to_list()

    for key, value, new_number in zip(X_dict.keys(), X_dict.values(), distance):
        X_dict[key] = value + (new_number,)


    X = list(X_dict.values())
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    Y = [i[0] for i in Y_dict.values()]
    Y = np.log10(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

    rf = RandomForestRegressor(n_estimators = 150, max_features = 'sqrt', max_depth = None, min_samples_leaf=2, min_samples_split=2, random_state = 18).fit(X_train, y_train)

    prediction = rf.predict(X_test)
    r2 = r2_score(y_test, prediction)
    print(r2)


def get_full_data():

        X_dict, Y_dict = load_everthing_old()
    distance  = pd.read_csv("data/distance_data/processed_distances.csv")
    distance["0"] = distance["0"].fillna(0)
    distance = distance['0'].to_list()

    for key, value, new_number in zip(X_dict.keys(), X_dict.values(), distance):
        X_dict[key] = value + (new_number,)

    return X_dict, Y_dict



def rf_without_distance():

    X_dict, Y_dict = load_everthing_old()
    X = list(X_dict.values())
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    Y = [i[0] for i in Y_dict.values()]
    Y = np.log10(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

   # rf = RandomForestRegressor(n_estimators = 150, max_features = 'sqrt', max_depth = None, min_samples_leaf=2, min_samples_split=2, random_state = 18).fit(X_train, y_train)
    rf = RandomForestRegressor().fit(X_train, y_train)

    prediction = rf.predict(X_test)
    r2 = r2_score(y_test, prediction)
    print(r2)



def rf_only_distance():


    distance = pd.read_csv("data/distance_data/processed_distances.csv")
    df_filled = distance.fillna(0)
    df_arr = df_filled.to_numpy().reshape(-1, 1)
    scaler = StandardScaler()
    X = scaler.fit_transform(df_arr)
    Y = [i[0] for i in Y_dict.values()]
    Y = np.log10(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
   # rf = RandomForestRegressor(n_estimators = 150, max_features = 'sqrt', max_depth = None, min_samples_leaf=2, min_samples_split=2, random_state = 18).fit(X_train, y_train)
    rf = RandomForestRegressor().fit(X_train, y_train)

    prediction = rf.predict(X_test)
    r2 = r2_score(y_test, prediction)
    print(r2)