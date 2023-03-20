import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def make_distribution_plots(metric, categories):
    # read in all the data files as a dictionary of data frames
    data_frames = {}
    for category in categories:
        file_path = f"data/fb_data/FB{metric}Dist_{category}.csv"
        df = pd.read_csv(file_path, index_col=0)
        data_frames[category] = df
    
    # Plotting each data frame side by side
    fig, axs = plt.subplots(3, 5, figsize=(17,10))
    for i, (category, df) in enumerate(data_frames.items()):
        axs[i // 5, i % 5].hist(df.values.flatten(), bins=70)
        axs[i // 5, i % 5].set_title(category)
    plt.show()

# Define the list of categories
categories = ["BusinessIndustry", "Education", "Uncategorized","FamilyRelationships","FitnessWellness","FoodDrink","HobbiesActivities","LifestyleCulture",'NewsEntertainment','NonLocalBus','People','ShoppingFashion','SportsOutdoors','Technology','TravelPlacesEvents']

# Calling the function with the list of categories
make_distribution_plots("Cos",categories)
# Metrics: Cos, Euc, Man, Het

# Standardized version
def std_make_distribution_plots(metric,categories):
    # read in all the data files as a dictionary of data frames
    data_frames = {}
    for category in categories:
        file_path = f"data/fb_data/FB{metric}Dist_{category}.csv"
        df = pd.read_csv(file_path, index_col=0)
        # standardize the data
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df)
        df = pd.DataFrame(df_scaled, index=df.index, columns=df.columns)
        data_frames[category] = df
    
    # Plotting each data frame side by side
    fig, axs = plt.subplots(3, 5, figsize=(17,10))
    for i, (category, df) in enumerate(data_frames.items()):
        axs[i // 5, i % 5].hist(df.values.flatten(), bins=70)
        axs[i // 5, i % 5].set_title(category)
    plt.show()

# Define the list of categories
categories = ["BusinessIndustry", "Education", "Uncategorized","FamilyRelationships","FitnessWellness","FoodDrink","HobbiesActivities","LifestyleCulture",'NewsEntertainment','NonLocalBus','People','ShoppingFashion','SportsOutdoors','Technology','TravelPlacesEvents']

# Calling the function with the list of categories
std_make_distribution_plots("Cos",categories)
# Metrics: Cos, Euc, Man, Het

