import pandas as pd
import os

def split_social_continent():
    # Reading in the data
    file_path = 'data/friendship_data/countries-countries-fb-social-connectedness-index-october-2021.tsv'
    df = pd.read_csv(file_path, sep='\t')
    
    # Loading the necessary country information
    country_info = pd.read_csv('src/Visualisations/Continents/all.csv', usecols=['alpha-2', 'region'], index_col='alpha-2')
    
    # Merging the country information with the main dataframe
    merged = df.merge(country_info, left_on='user_loc', right_index=True, suffixes=['_user', '_friend'])

    
    # Splitting the data into separate dataframes for each continent
    continents = merged['region'].unique()
    continent_dataframes = {}
    for continent in continents:
        continent_df = merged[(merged['region'] == continent)]
        continent_dataframes[continent] = continent_df
    
    # Saving each continent dataframe as a separate CSV file
    output_dir = 'data/friendship_data_continents'
    os.makedirs(output_dir, exist_ok=True)
    for continent, df in continent_dataframes.items():
        output_file = os.path.join(output_dir, f'{continent}.csv')
        df.to_csv(output_file, index=False)
    
    return continent_dataframes

continent_dfs = split_social_continent()

# Each dataframe by continent
asia = continent_dfs['Asia']
america = continent_dfs['Americas']
europe = continent_dfs['Europe']
oceania = continent_dfs['Oceania']
africa = continent_dfs['Africa']

print(africa)