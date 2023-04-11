import pandas as pd
from utils.load import load_iso_dict
import os
def create_tsv_labels(path):
    for file in os.listdir(path):
        if file.endswith("csv") == False:
            continue    
        with open(path + file) as f:
            f = f.readlines()
            s = ""
            for elm in f:
                elm = elm.replace(",","\t")
                s += elm
        file = file.replace("csv","tsv")
        f = open(f"data/friendship_data_continents_new/{file}","w")
        f.write(s)
        f.close()

def load_iso_codes(continent):
    """Loads the files and only gets the country 2 letter iso codes"""
    with open(f"data/iso_all/{continent}_iso_codes.txt", "r") as f:
        f = f.read()
        f = f.split("\n")
    li = []
    for elm in f:
        elm = elm.split("\t")
        li.append(elm[1])
    return li

def get_correct_countries(iso_path,df):
    with open(iso_path,"r") as f:
        f = f.readlines()
        iso_codes = [x.replace("\n","") for x in f]
    headers = list(set(df.columns) & set(iso_codes))
    df = df[(df["ISO_CODE"].isin(headers))]
    df = df[headers + ["ISO_CODE"]]
    df = df.reindex(sorted(df.columns), axis=1)
        # shift column 'ISO_CODE' to first position
    first_column = df.pop('ISO_CODE')
    # insert column using insert(position,column_name,first_column) function
    df.insert(0, 'ISO_CODE', first_column)
    return df

def gen_new_label_files():
    data = pd.read_csv("data/friendship_data/countries-countries-fb-social-connectedness-index-october-2021.tsv",delimiter="\t")
    iso = load_iso_dict()
    for key, val in iso.items():
        df = data[(data["user_loc"].isin(val)) & (data["fr_loc"].isin(val))]
        df = df.drop(["region"], axis=1, errors='ignore')
        df.to_csv(f"data/friendship_data_continents_new/{key}.csv",index = False)
