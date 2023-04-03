import pandas as pd
import os

# This function splits the data by continent
def continent(metric, category):
    # Reading in the data 
    data_frames = {}
    file_path = f"data/fb_data/FB{metric}Dist_{category}.csv"
    df = pd.read_csv(file_path, index_col=0)
    data_frames[category] = df    
    # Loading the necessary country information
    country_info = pd.read_csv('src/Visualisations/Continents/all.csv', usecols=['alpha-2', 'region'], index_col='alpha-2')
    
    # Merging the country information with the main dataframe
    merged = data_frames[category].merge(country_info, left_on='ISO_CODE', right_index=True).merge(country_info, left_on='ISO_CODE', right_index=True, suffixes=['_1', '_2'])
    
    # Splitting the data into separate dataframes for each continent
    continents = merged['region_1'].unique()
    continent_dataframes = {}
    for continent in continents:
        continent_df = merged[(merged['region_1'] == continent) & (merged['region_2'] == continent)]
        continent_dataframes[continent] = continent_df
    
    return continent_dataframes, metric, category

continent_dfs, metric, category = continent("Cos","Education")
# Metrics: Cos, Euc, Man, Het
# Categories = ["BusinessIndustry", "Education", "Uncategorized","FamilyRelationships","FitnessWellness","FoodDrink","HobbiesActivities","LifestyleCulture",'NewsEntertainment','NonLocalBus','People','ShoppingFashion','SportsOutdoors','Technology','TravelPlacesEvents']

# Each dataframe by continent
asia = continent_dfs['Asia']
america = continent_dfs['Americas']
europe = continent_dfs['Europe']
oceania = continent_dfs['Oceania']
africa = continent_dfs['Africa']

print(africa)
 
# The following function creastes CSV files of the continent dataframes in the folder fb_data_continents
'''
directory = "data/fb_data_continents"
if not os.path.exists(directory):
    os.makedirs(directory)

# Iterating over the dictionary of dataframes and saving each one to a CSV file
for continent, df in continent_dfs.items():
    filename = f"{metric}_{category}_{continent}.csv"
    filepath = os.path.join(directory, filename)
    df.to_csv(filepath, index=False)
    print(f"Saved {filename} with {len(df)} rows")
'''

# This following function does the job for all the data files 
'''
import os
import pandas as pd

# This function splits the data by continent
def split_by_continent(metric, category):
    # Reading in the data 
    file_path = f"data/fb_data/FB{metric}Dist_{category}.csv"
    df = pd.read_csv(file_path, index_col=0)
    
    # Loading the necessary country information
    country_info = pd.read_csv('src/Visualisations/Continents/all.csv', usecols=['alpha-2', 'region'], index_col='alpha-2')
    
    # Merging the country information with the main dataframe
    merged = df.merge(country_info, left_on='ISO_CODE', right_index=True).merge(country_info, left_on='ISO_CODE', right_index=True, suffixes=['_1', '_2'])
    
    # Splitting the data into separate dataframes for each continent
    continents = merged['region_1'].unique()
    continent_dataframes = {}
    for continent in continents:
        continent_df = merged[(merged['region_1'] == continent) & (merged['region_2'] == continent)]
        continent_dataframes[continent] = continent_df
    
    return continent_dataframes

# Defining the path where the data files are located and where to save them
data_path = "data/fb_data"
save_path = "data/fb_data_continents"

# Defining the metrics and categories
metrics = ['Cos', 'Euc', 'Man', 'Het']
categories = ["BusinessIndustry", "Education", "Uncategorized","FamilyRelationships","FitnessWellness","FoodDrink","HobbiesActivities","LifestyleCulture",'NewsEntertainment','NonLocalBus','People','ShoppingFashion','SportsOutdoors','Technology','TravelPlacesEvents']

# Looping through all the data files
for metric in metrics:
    for category in categories:
        filename = f"FB{metric}Dist_{category}.csv"
        file_path = os.path.join(data_path, filename)

        # Splitting the data by continent
        continent_dataframes = split_by_continent(metric, category)

        # Saving the dataframes to csv files for each continent
        for continent, df in continent_dataframes.items():
            df.to_csv(os.path.join(save_path, f"{filename}_{continent}.csv"), index=False)
'''