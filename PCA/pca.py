
import sys
sys.path.append('..')
import os
import pandas as pd
import lib
import numpy as np

print(lib.make_total_df())

"""
label_names, taget_data = lib.create_target_data()


X_dict, Y_dict =lib.load_everthing()

X = list(X_dict.values())
Y = [x[0] for x in Y_dict.values()]

labels = np.reshape(Y,(len(Y),1))
data = np.concatenate([X,labels],axis=1)
df = pd.DataFrame(data)
feature_names = lib.get_feature_names()
#feature_names.insert(0,)

feature_names.append("Y_labels")
df.columns = feature_names

print(df.head())

"""