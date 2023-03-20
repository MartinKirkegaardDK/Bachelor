import sys
sys.path.append('..')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def dist_labels(filepath, countrycode):
    # Read in the data
    df = pd.read_csv(filepath, delimiter= "\t",keep_default_na=False)

    # Defining the country of interest
    country = countrycode

    # Filtering the data to include the country of interest and all other countries
    df_filtered = df.loc[((df["user_loc"] == country) & (df["fr_loc"] != country)) | ((df["user_loc"] != country) & (df["fr_loc"] == country))].copy()

    # Calculating the standardization for the entire dataset
    scaler = StandardScaler()
    df_filtered["scaled_sci"] = scaler.fit_transform(df_filtered[["scaled_sci"]])

    # Take the log of the scaled_sci column
    df_filtered["log_scaled_sci"] = np.log(df_filtered["scaled_sci"])

    # Plot the distribution of log_scaled_sci for the country of interest as user_loc
    sns.histplot(data=df_filtered, x="log_scaled_sci", kde=True)
    plt.title(f"Distribution of log scaled_sci for {country} as user_loc")
    plt.show()

dist_labels("../../data/friendship_data/countries-countries-fb-social-connectedness-index-october-2021.tsv","SE")