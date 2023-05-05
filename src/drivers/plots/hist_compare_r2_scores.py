
from sklearn.linear_model import LassoCV
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils.load import load_everthing_old
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def rf():

    labels = ['All data with distance', 'All data without distance', 'Distance']
    values = [0.71, 0.65, 0.18]
    barWidth = 0.5
    r1 = np.arange(len(values))
    plt.bar(r1, values, width=barWidth, color='blue')

    plt.xticks([r for r in range(len(values))], labels)

    # Set plot title and axis labels
    plt.title('R-squared score for rf all data with and without distance and just distance')
    plt.xlabel('Dataset')
    plt.ylabel('R-squared Score')

    # Show the plot
    plt.savefig('plots/r2_comparisons_rf.png')




def train_lr_only_dist():
    distance = pd.read_csv("data/distance_data/processed_distances.csv")
    X_dict, Y_dict = load_everthing_old()

    df_filled = distance.fillna(0)
    df_arr = df_filled.to_numpy().reshape(-1, 1)
    scaler = StandardScaler()
    X = scaler.fit_transform(df_arr)
    Y = [i[0] for i in Y_dict.values()]
    Y = np.log10(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)



    lr = LassoCV(max_iter= 100000, tol= 0.001).fit(X_train, y_train)

    prediction = lr.predict(X_test)
    r2 = r2_score(y_test, prediction)

    return r2


def lf():

    labels = ['All data with distance', 'All data without distance', 'Distance']
    dist_r2 = train_lr_only_dist()
    values = [0.62, 0.56, dist_r2]
    barWidth = 0.5
    r1 = np.arange(len(values))
    plt.bar(r1, values, width=barWidth, color='blue')

    plt.xticks([r for r in range(len(values))], labels)

    # Set plot title and axis labels
    plt.title('R-squared score for lr all data with and without distance and just distance')
    plt.xlabel('Dataset')
    plt.ylabel('R-squared Score')

    # Show the plot
    plt.savefig('plots/r2_comparisons_lr.png')
