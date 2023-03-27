


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.load import make_total_df,get_feature_names, get_indv_df

LOCAL_DATA = False 

if  LOCAL_DATA:
    data = pd.read_csv('combined_df.csv')
    df = pd.DataFrame(data)
    print("test")

else:
    df = make_total_df()

def pca_func():

    df = make_total_df()

    feature_names = get_feature_names('cos')
    x = df.loc[:, feature_names].values
    x = StandardScaler().fit_transform(x) # normalizing the features
    x = pd.DataFrame(x, columns=feature_names)
    feat_cols = ['feature'+str(i) for i in range(x.shape[1])]
    normalised_data = pd.DataFrame(x,columns=feat_cols)
    pcamodel = PCA(n_components=5)
    pca = pcamodel.fit_transform(x)
    return x,pca, pcamodel


def scree_plot():
    x, pca,pcamodel = pca_func()
    plt.plot(pcamodel.explained_variance_)
    plt.title('Scree Plot')
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
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


#cos_df = get_indv_df('cos')

#x, pca, pcamodel = pca_func(cos_df)

#myplot(pca[:,0:2],np.transpose(pcamodel.components_[0:2, :]),list(x.columns))
#plt.show()




