import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def dist_plots(metric, categories):
    # Reading in the data as a dictionary of dataframes
    data_frames = {}
    for category in categories:
        file_path = f"data/fb_data/FB{metric}Dist_{category}.csv"
        df = pd.read_csv(file_path, index_col=0)
        data_frames[category] = df
    
    # Plotting the dataframes
    fig, axs = plt.subplots(3, 5, figsize=(17,10))
    for i, (category, df) in enumerate(data_frames.items()):
        axs[i // 5, i % 5].hist(df.values.flatten(), bins=70)
        axs[i // 5, i % 5].set_title(category)
    plt.show()

# Defining the list of categories (features)
categories = ["BusinessIndustry", "Education", "Uncategorized","FamilyRelationships","FitnessWellness","FoodDrink","HobbiesActivities","LifestyleCulture",'NewsEntertainment','NonLocalBus','People','ShoppingFashion','SportsOutdoors','Technology','TravelPlacesEvents']

# Calling the function with the list of categories
dist_plots("Cos",categories)
# Metrics: Cos, Euc, Man, Het

# Standardized version
def std_dist_plots(metric,categories):
    # Reading in the data as a dictionary of dataframes
    data_frames = {}
    for category in categories:
        file_path = f"data/fb_data/FB{metric}Dist_{category}.csv"
        df = pd.read_csv(file_path, index_col=0)
        # Standardizing
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df)
        df = pd.DataFrame(df_scaled, index=df.index, columns=df.columns)
        data_frames[category] = df
    
    # Plotting
    fig, axs = plt.subplots(3, 5, figsize=(17,10))
    for i, (category, df) in enumerate(data_frames.items()):
        axs[i // 5, i % 5].hist(df.values.flatten(), bins=70)
        axs[i // 5, i % 5].set_title(category)
    plt.show()

# Calling the function with the list of categories
std_dist_plots("Cos",categories)


