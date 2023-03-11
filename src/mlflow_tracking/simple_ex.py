import sys
sys.path.append('..')
import os
import sklearn
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.svm import SVC
import numpy as np
from sklearn.linear_model import LinearRegression
import lib 

friendship_data = pd.read_csv("../../data/friendship_data/countries-countries-fb-social-connectedness-index-october-2021.tsv", delimiter= "\t",keep_default_na=False)

friendship_data = lib.preprocess(friendship_data)

codes = list(friendship_data["ISO_CODE"])
cartesian_df = lib.create_cartesian_product(codes)
cartesian_df.insert(0,'ISO_CODES',codes)
cartesian_df = lib.select_relevant_cells(cartesian_df)
label_names = lib.from_diagonal_to_list(cartesian_df)
friendship_data = lib.df_to_list(friendship_data)

fb_data_technology = lib.load("../fb_data/FBCosDist_Technology.csv",",")
fb_data_technology = lib.preprocess(fb_data_technology)
fb_data_technology = lib.df_to_list(fb_data_technology)

fb_data_business = lib.load("../fb_data/FBCosDist_BusinessIndustry.csv",",")
fb_data_business = lib.preprocess(fb_data_business)
fb_data_business = lib.df_to_list(fb_data_business)




X_list = [fb_data_technology,fb_data_business]
Y_list = [friendship_data]
X_dict = lib.create_label_dict(label_names,X_list)
Y_dict = lib.create_label_dict(label_names, Y_list)


with mlflow.start_run():

    X = list(X_dict.values())
    y = list(Y_dict.values())
    reg = LinearRegression().fit(X, y)
    score = reg.score(X, y)

    print("Our score: ",reg.score(X, y))

    mlflow.log_metric("score", score)
    mlflow.sklearn.log_model(reg, "model")
