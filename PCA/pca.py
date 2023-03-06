
import sys
sys.path.append('..')
import lib
from sklearn.preprocessing import StandardScaler
import numpy as np


X_dict, Y_dict = lib.load_everthing()
X = list(X_dict.values())

df = lib.make_total_df()
x = df.loc[:, X].values
x = StandardScaler().fit_transform(x) # normalizing the features


print(x.shape)
print(np.mean(x),np.std(x))




