import pandas as pd
#from utils.load import make_total_df
import country_converter as coco
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt



def preprocess():

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

    distance_df = distance_df[['iso_o', 'dist']]

    friendship_data = pd.read_csv("data/friendship_data/countries-countries-fb-social-connectedness-index-october-2021.tsv", delimiter="\t")
    friendship_data['scaled_sci'] = np.log2(friendship_data['scaled_sci']+1)
    #friendship_data = friendship_data.round({'scaled_sci'})
    friendship_data["user_loc"] = friendship_data["user_loc"] + '_' + friendship_data["fr_loc"]

    friendship_data['friend_rank'] = range(1, len(friendship_data) + 1)
    f_df = friendship_data.sort_values(by="scaled_sci", ascending=False)
    distance_df['dist_rank'] = range(1, len(distance_df) + 1)
    d_df = distance_df.sort_values(by="dist", ascending=False)

    df = pd.merge(left=f_df, right=d_df, how='left', left_on='user_loc', right_on='iso_o')
    df = df.dropna()


    return df




# Used this: https://itnext.io/boost-your-data-science-with-ranking-in-python-and-pandas-911080128e1
def plot_with_title(print_spearman_correlation=True):
   
    df = preprocess()

    x = df['friend_rank']
    y = df['dist_rank']
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(x, y)
    # construct title
    title = f"Correlation {pearsonr(x, y)[0]:.3f}"
    if print_spearman_correlation:
        title += f", Spearman correlation {spearmanr(x, y)[0]:.3f}"
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    plt.show()
    
