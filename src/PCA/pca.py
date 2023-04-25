


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.load import make_total_df,get_feature_names, get_indv_df


df = make_total_df()

def pca_func():

    df = make_total_df()

    feature_names = get_feature_names('cos')
    x = df.loc[:, feature_names].values
    x = StandardScaler().fit_transform(x) # normalizing the features
    x = pd.DataFrame(x, columns=feature_names)
    feat_cols = ['feature'+str(i) for i in range(x.shape[1])]
    normalised_data = pd.DataFrame(x,columns=feat_cols)
    pcamodel = PCA(n_components=7)
    pca = pcamodel.fit_transform(x)

    return x,pca, pcamodel


def pca_improve_data():


    feature_names = get_feature_names('cos')
    x = df.loc[:, feature_names].values
    x = StandardScaler().fit_transform(x) # normalizing the features
    mean_vec = np.mean( x, axis=0)
    cov_mat = np.cov( x.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    # Create a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [ (np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]

    # Sort the eigenvalue, eigenvector pair from high to low
    eig_pairs.sort(key = lambda x: x[0], reverse= True)

    # Calculation of Explained Variance from the eigenvalues
    tot = sum(eig_vals)
    var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)] # Individual explained variance
    cum_var_exp = np.cumsum(var_exp) # Cumulative explained variance

    res = [n for n,i in enumerate(cum_var_exp) if i>90 ][0]  # Amount of vectors needed to explain 90 percent of the variance
    pca = PCA(n_components=res)
    pca.fit(x)
    X_res = pca.transform(x)

    return X_res, df


def pca_random_forest():

    X_res, df = pca_improve_data()
    Y = df["Y_labels"]
    Y = np.log10(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

    rf = RandomForestRegressor(n_estimators = 150, max_features = 'sqrt', max_depth = None, min_samples_leaf=2, min_samples_split=2, random_state = 18).fit(X_train, y_train)

    prediction = rf.predict(X_test)
    r2 = r2_score(y_test, prediction)
    print(r2)




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
    plt.scatter(xs * scalex,ys * scaley,s=2,edgecolors='none')
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1, coeff[i,1] * 1, "Var"+str(i+1), color = 'green', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1, coeff[i,1] * 1, labels[i], color = 'green', ha = 'center', va = 'center')
 
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()
    plt.show()


#cos_df = get_indv_df('cos')

#x, pca, pcamodel = pca_func(cos_df)

#myplot(pca[:,0:2],np.transpose(pcamodel.components_[0:2, :]),list(x.columns))
#plt.show()




