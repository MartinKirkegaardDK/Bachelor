
# Custom libraries
from utils.load import make_total_df, get_feature_names, get_indv_df

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor 

import matplotlib.pyplot as plt
import pandas as pd

def make_pcr_features(n_components):

    # Load in the data
    df = make_total_df()
   # df = df.drop("Y_labels", axis=1)

    distance = pd.read_csv("data/distance_data/processed_distances.csv")
    df_filled = distance.fillna(0)
    df_arr = df_filled.to_numpy().reshape(-1, 1)

    # Choose the columns with all distance metrics
    feature_names = get_feature_names('cos') + get_feature_names('euc') + get_feature_names('man') + get_feature_names('het')
    x = df.loc[:, feature_names].values 
    x = np.concatenate((x, df_arr), axis=1)

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
    pca = PCA(n_components=n_components)
    pca.fit(x)
    X = pca.transform(x)

    Y = df["Y_labels"]
    Y = np.log10(Y)

    return X, Y



def scree_plot_for_indv_metrics():

    X, Y = make_pcr_features(15)
    metrics = ['cos','euc','man','het']

    for met in range(4):
        print(metrics[met])
        pcas = []
        r2_scores = []

        for i in range(1,15):
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
            rf = RandomForestRegressor(n_estimators = 150, max_features = 'sqrt', max_depth = None, min_samples_leaf=2, min_samples_split=2, random_state = 18).fit(X_train, y_train)
            prediction = rf.predict(X_test)
            r2 = r2_score(y_test, prediction)
            pcas.append(i)
            r2_scores.append(r2)

        plt.plot(pcas, r2_scores)

        # Add axis labels and a title to the plot
        plt.xlabel('Number of Principal Components')
        plt.ylabel('R-squared Value')
        plt.title('Number of Principal Components vs. R-squared value')

        # Show the plot
        plt.show()




def scree_plot_all_data():

    X, Y = make_pcr_features(40)
    pcas = []
    r2_scores = []
    for i in range(1,40):

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
        rf = RandomForestRegressor(n_estimators = 150, max_features = 'sqrt', max_depth = None, min_samples_leaf=2, min_samples_split=2, random_state = 18).fit(X_train, y_train)
        prediction = rf.predict(X_test)
        r2 = r2_score(y_test, prediction)
        pcas.append(i)
        r2_scores.append(r2)

    plt.plot(pcas, r2_scores)

    # Add axis labels and a title to the plot
    plt.xlabel('Number of Principal Components')
    plt.ylabel('R-squared Value')
    plt.title('Number of Principal Components vs. R-squared value')

    # Show the plot
    plt.show()


def rf_with_distance():

    X_dict, Y_dict = load_everthing_old()
    distance  = pd.read_csv("data/distance_data/processed_distances.csv")
    distance["0"] = distance["0"].fillna(0)
    distance = distance['0'].to_list()

    for key, value, new_number in zip(X_dict.keys(), X_dict.values(), distance):
        X_dict[key] = value + (new_number,)


    X = list(X_dict.values())
    Y = [i[0] for i in Y_dict.values()]
    Y = np.log10(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

    rf = RandomForestRegressor(n_estimators = 150, max_features = 'sqrt', max_depth = None, min_samples_leaf=2, min_samples_split=2, random_state = 18).fit(X_train, y_train)

    prediction = rf.predict(X_test)
    r2 = r2_score(y_test, prediction)
    print(r2)




def rf_without_distance():

    X_dict, Y_dict = load_everthing_old()
    X = list(X_dict.values())
    Y = [i[0] for i in Y_dict.values()]
    Y = np.log10(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

    rf = RandomForestRegressor(n_estimators = 150, max_features = 'sqrt', max_depth = None, min_samples_leaf=2, min_samples_split=2, random_state = 18).fit(X_train, y_train)

    prediction = rf.predict(X_test)
    r2 = r2_score(y_test, prediction)
    print(r2)