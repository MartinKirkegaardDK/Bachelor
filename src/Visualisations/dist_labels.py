import sys
sys.path.append('..')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def dist_labels(file_path):
    df = pd.read_csv(file_path, delimiter= "\t",keep_default_na=False)

    # Standardize the data
    scaler = StandardScaler()
    df["scaled_sci"] = scaler.fit_transform(df[["scaled_sci"]])

    # Plot the distribution of the scaled_sci column
    sns.histplot(data=df, x="scaled_sci", kde=True)
    plt.show()


dist_labels("../../data/friendship_data/countries-countries-fb-social-connectedness-index-october-2021.tsv")