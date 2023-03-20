import sys
sys.path.append('..')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Plotting each 
def make_distribution_plots(categories):
    # read in all the data files as a dictionary of data frames
    data_frames = {}
    for category in categories:
        file_path = f"../../data/fb_data/FBManDist_{category}.csv"
        #file_path = f"../../data/fb_data/FBCosDist_{category}.csv"
        #file_path = f"../../data/fb_data/FBEucDist_{category}.csv"
        #file_path = f"../../data/fb_data/FBHetDist_{category}.csv"
        df = pd.read_csv(file_path, index_col=0)
        data_frames[category] = df
    
    # Plotting each data frame side by side
    fig, axs = plt.subplots(3, 5, figsize=(17,10))
    for i, (category, df) in enumerate(data_frames.items()):
        axs[i // 5, i % 5].hist(df.values.flatten(), bins=70)
        axs[i // 5, i % 5].set_title(category)
        #axs[i // 5, i % 5].set_xlabel('Value')
    plt.show()

# Define the list of categories
categories = ["BusinessIndustry", "Education", "Uncategorized","FamilyRelationships","FitnessWellness","FoodDrink","HobbiesActivities","LifestyleCulture",'NewsEntertainment','NonLocalBus','People','ShoppingFashion','SportsOutdoors','Technology','TravelPlacesEvents']

# Calling the function with the list of categories
make_distribution_plots(categories)
'''

import sys
sys.path.append('..')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Plotting each 
def make_distribution_plots(categories):
    # read in all the data files as a dictionary of data frames
    data_frames = {}
    for category in categories:
        #file_path = f"../../data/fb_data/FBManDist_{category}.csv"
        #file_path = f"../../data/fb_data/FBCosDist_{category}.csv"
        file_path = f"../../data/fb_data/FBEucDist_{category}.csv"
        #file_path = f"../../data/fb_data/FBHetDist_{category}.csv"
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
        #axs[i // 5, i % 5].set_xlabel('Value')
    plt.show()

# Define the list of categories
categories = ["BusinessIndustry", "Education", "Uncategorized","FamilyRelationships","FitnessWellness","FoodDrink","HobbiesActivities","LifestyleCulture",'NewsEntertainment','NonLocalBus','People','ShoppingFashion','SportsOutdoors','Technology','TravelPlacesEvents']

# Calling the function with the list of categories
make_distribution_plots(categories)
