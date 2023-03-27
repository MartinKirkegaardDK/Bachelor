import pandas as pd
import numpy as np

# This finds and prints the (if any) non-positive values 

def check_non_positive_values(metric,categories):
    data_frames = {}
    for category in categories:
        file_path = f"data/fb_data/FB{metric}Dist_{category}.csv"
        df = pd.read_csv(file_path, index_col=0)
        data_frames[category] = df
        non_positive_values = df[df <= 0].dropna()
        if not non_positive_values.empty:
            print(f"The following non-positive values were found for {metric} for {category}:\n{non_positive_values}")
        else:
            print(f"No non-positive values were found for {metric} for {category}.")

# Define the list of categories
categories = ["BusinessIndustry", "Education", "Uncategorized","FamilyRelationships","FitnessWellness","FoodDrink","HobbiesActivities","LifestyleCulture",'NewsEntertainment','NonLocalBus','People','ShoppingFashion','SportsOutdoors','Technology','TravelPlacesEvents']

check_non_positive_values("Cos",categories)
check_non_positive_values("Euc",categories)
check_non_positive_values("Man",categories)
check_non_positive_values("Het",categories)