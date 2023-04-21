

import pandas as pd
import numpy as np
import country_converter as coco
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from utils.load import make_total_df



def make_X_data():

    distance_df = pd.read_excel('data/distance_data/dist_cepii.xls')

    # Changing country codes from iso3 to iso2
    iso3_codes = distance_df['iso_o'].to_list()
    iso2_codes_list = coco.convert(names=iso3_codes, to='ISO2', not_found='NULL')
    distance_df['iso_o'] = iso2_codes_list
    distance_df = distance_df.drop(distance_df.loc[distance_df['iso_o'] == 'NULL'].index)

    iso3_codes = distance_df['iso_d'].to_list()
    iso2_codes_list = coco.convert(names=iso3_codes, to='ISO2', not_found='NULL')
    distance_df['iso_d'] = iso2_codes_list
    distance_df = distance_df.drop(distance_df.loc[distance_df['iso_d'] == 'NULL'].index)
    distance_df["iso_o"] = distance_df["iso_o"] + '_' + distance_df["iso_d"]

    distance_df = distance_df[['iso_o', 'dist']]

    total_df = make_total_df()
    total_df = total_df.rename_axis('c_to_c').reset_index()
    total_df.head()

    df = pd.merge(left=total_df, right=distance_df, how='left', left_on='c_to_c', right_on='iso_o')
    df = df.drop(['iso_o', 'Y_labels'], axis=1)


    distance = df["dist"].to_list()

    df = pd.DataFrame(distance)
    csv_data = df.to_csv(index=False)
    df.to_csv('data/distance_data/processed_distances.csv', index=False)

    return df



def make_Y_data():

    friendship_data = pd.read_csv("data/friendship_data/countries-countries-fb-social-connectedness-index-october-2021.tsv", delimiter="\t")
    friendship_data['scaled_sci'] = np.log2(friendship_data['scaled_sci']+1)
    #friendship_data = friendship_data.round({'scaled_sci'})
    friendship_data["user_loc"] = friendship_data["user_loc"] + '_' + friendship_data["fr_loc"]

    plot_df = pd.merge(left=friendship_data, right=distance_df, how='left', left_on='user_loc', right_on='iso_o')
    plot_df = plot_df.dropna()

    return plot_df



