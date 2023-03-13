import sys
sys.path.insert(0, '..')
from config.definitions import ROOT_DIR
import os
file_path = os.path.join(ROOT_DIR, 'data')

import pandas as pd

friendship_data = pd.read_csv("../data/friendship_data/countries-countries-fb-social-connectedness-index-october-2021.tsv",delimiter='\t')

def check_link_values(country1, country2, data):
    df1 = data[(data['user_loc'] == country1) & (data['fr_loc'] == country2)]
    df2 = data[(data['user_loc'] == country2) & (data['fr_loc'] == country1)]
    if len(df1) == 0 and len(df2) == 0:
        return "No links between the countries found."
    elif len(df1) == 0 and len(df2) == 1:
        return f"Only one link found: {country2} -> {country1} = {df2.iloc[0]['scaled_sci']}"
    elif len(df2) == 0 and len(df1) == 1:
        return f"Only one link found: {country1} -> {country2} = {df1.iloc[0]['scaled_sci']}"
    elif len(df1) == 1 and len(df2) == 1:
        if df1.iloc[0]['scaled_sci'] == df2.iloc[0]['scaled_sci']:
            return f"The links between the countries have the same value: {country1} -> {country2} = {df1.iloc[0]['scaled_sci']}, {country2} -> {country1} = {df2.iloc[0]['scaled_sci']}"
        else:
            return f"The links between the countries have different values: {country1} -> {country2} = {df1.iloc[0]['scaled_sci']}, {country2} -> {country1} = {df2.iloc[0]['scaled_sci']}"
    else:
        return "Unexpected error occurred."

result = check_link_values('DE','DK',friendship_data)
print(result)
