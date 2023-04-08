import pandas as pd
from utils.load import make_total_df
import country_converter as coco
import numpy as np



def make_dfs():

    distance_df = pd.read_excel('dist_cepii.xls')

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


    distance_df["iso_o"] = distance_df["iso_o"] + '_' + distance_df["iso_d"]
    distance_df = distance_df[['iso_o', 'dist']]

    friendship_data = pd.read_csv("data/friendship_data/countries-countries-fb-social-connectedness-index-october-2021.tsv", delimiter="\t")
    friendship_data['scaled_sci'] = np.log2(friendship_data['scaled_sci']+1)
    #friendship_data = friendship_data.round({'scaled_sci'})
    friendship_data["user_loc"] = friendship_data["user_loc"] + '_' + friendship_data["fr_loc"]

    return friendship_data, distance_df


def test():

    friendship_data, distance_df = make_dfs()

    df = pd.merge(left=friendship_data, right=distance_df, how='left', left_on='user_loc', right_on='iso_o')
