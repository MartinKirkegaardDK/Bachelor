import sys
sys.path.append('..')
import lib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Plotting each 
def make_distribution_plots(categories):
    # read in all the data files as a dictionary of data frames
    data_frames = {}
    for category in categories:
        file_path = f"../data/fb_data/FBManDist_{category}.csv"
        #file_path = f"../data/fb_data/FBCosDist_{category}.csv"
        #file_path = f"../data/fb_data/FBEucDist_{category}.csv"
        #file_path = f"../data/fb_data/FBHetDist_{category}.csv"
        df = pd.read_csv(file_path, index_col=0)
        data_frames[category] = df
    
    # Plotting each data frame side by side
    fig, axs = plt.subplots(1, len(data_frames), figsize=(17,7))
    for i, (category, df) in enumerate(data_frames.items()):
        axs[i].hist(df.values.flatten(), bins=50)
        axs[i].set_title(category,rotation = 90)
        axs[i].set_xlabel('Value')
        #axs[i].set_ylabel('Frequency')
    plt.show()

# Define the list of categories
categories = ["BusinessIndustry", "Education", "Empty","FamilyRelationships","FitnessWellness","FoodDrink","HobbiesActivities","LifestyleCulture",'NewsEntertainment','NonLocalBus','People','ShoppingFashion','SportsOutdoors','Technology','TravelPlacesEvents']

# Calling the function with the list of categories
make_distribution_plots(categories)