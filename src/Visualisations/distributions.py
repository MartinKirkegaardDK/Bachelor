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
        axs[i // 5, i % 5].hist(df.values.flatten(),bins=70)
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



def julie_dist(df, metric):

    l = list(df.columns)

    # Getting the names of each category
    clean = []
    for i in l:
        i = i.split("_")
        clean.append(i[1].strip('.'))

    fig, axes = plt.subplots(3,5, figsize=(20, 10))
    ax = axes.flatten()

    columns = []
    for i in df.columns: # Getting only the columns with the metric that we're looking for
        if metric in i:
            columns.append(i)

    for i, col in enumerate(columns):
        sns.histplot(df[col],ax=ax[i],color = 'navy', kde = True) #Make evert individual histogram
        ax[i].set_title(clean[i], fontdict={'size': 10, 'weight': 'bold'})
        ax[i].set(xlabel='Count')
        
        sns.despine()

    plt.subplots_adjust(top=0.9) 
    plt.suptitle(f"Distributions of each category using {metric} distance" )
    fig.tight_layout() # change padding 


    plt.show()