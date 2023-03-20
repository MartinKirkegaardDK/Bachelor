import sys
sys.path.append('..')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def dist_labels(filepath, countrycode, number):
    # Read in the data
    df = pd.read_csv(filepath, delimiter="\t", keep_default_na=False)

    # Defining the country of interest
    country = countrycode

    # Filter the data to only include relations where the country of interest is the user_loc
    relations = df[df["user_loc"] == country]

    # Filter out self-loops
    relations = relations.loc[relations['fr_loc'] != country]

    # Sort the relations by scaled_sci values and select the top 10 and bottom 10
    top_relations = relations.nlargest(number, "scaled_sci")
    bottom_relations = relations.nsmallest(number, "scaled_sci")

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the top 10 relations
    axs[0].bar(top_relations["fr_loc"], top_relations["scaled_sci"])
    axs[0].set_xlabel("Country")
    axs[0].set_ylabel("Scaled Sci Value")
    axs[0].set_title(f"Top {number} relations for {country}")
    axs[0].tick_params(axis='x', labelrotation=90)

    # Plot the bottom 10 relations
    axs[1].bar(bottom_relations["fr_loc"], bottom_relations["scaled_sci"])
    axs[1].set_xlabel("Country")
    axs[1].set_ylabel("Scaled Sci Value")
    axs[1].set_title(f"Bottom {number} relations for {country}")
    axs[1].tick_params(axis='x', labelrotation=90)

    # Show the plot
    plt.show()

dist_labels("../../data/friendship_data/countries-countries-fb-social-connectedness-index-october-2021.tsv","DK",15)