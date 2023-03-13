

"""
import os
import sys

# Add root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import definitions module
from config import definitions
data_path = os.path.join(definitions.ROOT_DIR, 'data')
"""
import os
import sys
#sys.path.insert(0, os.path.abspath(''))
#from config.definitions import ROOT_DIR

#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#sys.path.insert(0,'..')
#import utils.lib as lib

#data_path = os.path.join(ROOT_DIR)


"""
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOCAL_DATA = False 

if  LOCAL_DATA:
    data = pd.read_csv('combined_df.csv')
    df = pd.DataFrame(data)
    print("test")

else:
    df = lib.make_total_df()

def pca_func(df):

    feature_names = lib.get_feature_names('cos')
    x = df.loc[:, feature_names].values
    x = StandardScaler().fit_transform(x) # normalizing the features
    x = pd.DataFrame(x, columns=feature_names)
    feat_cols = ['feature'+str(i) for i in range(x.shape[1])]
    normalised_data = pd.DataFrame(x,columns=feat_cols)
    pcamodel = PCA(n_components=5)
    pca = pcamodel.fit_transform(x)
    return x,pca, pcamodel


def scree_plot(df):
    x, pca,pcamodel = pca_func(df)
    plt.plot(pcamodel.explained_variance_)
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()




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


cos_df = lib.get_indv_df('cos')

x, pca, pcamodel = pca_func(cos_df)

myplot(pca[:,0:2],np.transpose(pcamodel.components_[0:2, :]),list(x.columns))
plt.show()

"""


