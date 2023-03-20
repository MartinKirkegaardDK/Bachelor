import sys
sys.path.append('..')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def top_dist_labels(filepath, countrycode, number):
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

top_dist_labels("data/friendship_data/countries-countries-fb-social-connectedness-index-october-2021.tsv","DK",15)

def logged_top_dist_labels(filepath, countrycode, number):
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

    # Log-scale transforming the scaled_sci values
    top_relations["scaled_sci"] = np.log10(top_relations["scaled_sci"])
    bottom_relations["scaled_sci"] = np.log10(bottom_relations["scaled_sci"])

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

logged_top_dist_labels("data/friendship_data/countries-countries-fb-social-connectedness-index-october-2021.tsv","DK",10)   

def all(filepath, countrycode):
     # Read in the data
    df = pd.read_csv(filepath, delimiter="\t", keep_default_na=False)

    # Defining the country of interest
    country = countrycode

    # Filter the data to only include relations where the country of interest is the user_loc
    relations = df[df["user_loc"] == country]

    # Filter out self-loops
    relations = relations.loc[relations['fr_loc'] != country]

    # Sort the relations by scaled_sci values in descending order
    relations = relations.sort_values(by="scaled_sci", ascending=False)

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot the top 10 relations
    ax.bar(relations["fr_loc"], relations["scaled_sci"])
    ax.set_xlabel("Country")
    ax.set_ylabel("Scaled Sci Value")
    ax.set_title(f"Relations for {country} (not log-scale transformed)")
    ax.tick_params(axis='x', labelrotation=90)

    # Show the plot
    plt.show()

all("data/friendship_data/countries-countries-fb-social-connectedness-index-october-2021.tsv","DK") 

def logged_all(filepath, countrycode):
     # Read in the data
    df = pd.read_csv(filepath, delimiter="\t", keep_default_na=False)

    # Defining the country of interest
    country = countrycode

    # Filter the data to only include relations where the country of interest is the user_loc
    relations = df[df["user_loc"] == country]

    # Filter out self-loops
    relations = relations.loc[relations['fr_loc'] != country]

    # Sort the relations by scaled_sci values in descending order
    relations = relations.sort_values(by="scaled_sci", ascending=False)

    # Log-scale transforming the scaled_sci values
    relations["scaled_sci"] = np.log10(relations["scaled_sci"])

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot the top 10 relations
    ax.bar(relations["fr_loc"], relations["scaled_sci"])
    ax.set_xlabel("Country")
    ax.set_ylabel("Scaled Sci Value")
    ax.set_title(f"Relations for {country} (log-scale transformed)")
    ax.tick_params(axis='x', labelrotation=90)

    # Show the plot
    plt.show()

logged_all("data/friendship_data/countries-countries-fb-social-connectedness-index-october-2021.tsv","DK")   


