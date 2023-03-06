
import sys
sys.path.append('..')
import PCA.lib as lib
import sklearn
from sklearn.preprocessing import StandardScaler
import numpy as np


X_dict, Y_dict = lib.load_everthing()
X = list(X_dict.values())

df = lib.make_total_df()
x = df.loc[:, X].values
x = StandardScaler().fit_transform(x) # normalizing the features


print(x.shape)
print(np.mean(x),np.std(x))






sys.path.append('..')
import lib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#X_dict, Y_dict = lib.load_everthing()
#X = list(X_dict.values())

X_dict, Y_dict = lib.load_everthing()
X = list(X_dict.values())
LOCAL_DATA = True # Reviews and models are downloaded if True

df = lib.make_total_df()
x = df.loc[:, X].values

if  LOCAL_DATA:
    df = pd.read_csv('combined_df.csv')

else:
    df = lib.make_total_df()

print(df)




"""
feature_names = lib.get_feature_names()
x = df.loc[:, feature_names].values
x = StandardScaler().fit_transform(x) # normalizing the features
x = pd.DataFrame(x, columns=feature_names)
feat_cols = ['feature'+str(i) for i in range(x.shape[1])]
normalised_data = pd.DataFrame(x,columns=feat_cols)
pcamodel = PCA(n_components=5)
pca = pcamodel.fit_transform(x)




plt.plot(pcamodel.explained_variance_ratio_)
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()


plt.plot(pcamodel.explained_variance_)
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

print(x.shape)
print(np.mean(x),np.std(x))
def myplot(score,coeff,labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley,s=5)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'green', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
 
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()

myplot(pca[:,0:2],np.transpose(pcamodel.components_[0:2, :]),list(x.columns))
plt.show()



"""