import pandas as pd
import warnings
import numpy as np
#from pathlib import Path
import sys

import os
from config.definitions import ROOT_DIR



def test():
    print("hejsa")

def remove_self_loops(pandas_dataframe):
    """Removes self loops from a pandas dataframe by removing where two locations are equal. Example: AE = AE gets removed. Removing 184 selfloops"""
    mask = pandas_dataframe.iloc[:,0] != pandas_dataframe.iloc[:,1]
    return pandas_dataframe[mask]

def get_common(fb_data, friendship_data):
    """gets the intersection of ISO-codes between the two datasets. fb_data.shape should return (33672, 3)"""
    facebook_uniques = set(list(fb_data["ISO_CODE_1"].unique()))
    friendship_uniques = set(list(friendship_data["user_loc"].unique()))
    result = facebook_uniques.intersection(friendship_uniques)
    fb_data = fb_data[(fb_data["ISO_CODE_1"].isin(result)) & (fb_data["ISO_CODE_2"].isin(result))]
    friendship_data = friendship_data[(friendship_data["user_loc"].isin(result)) & (friendship_data["fr_loc"].isin(result))]
    return fb_data, friendship_data

def get_intersection():
    """Gets all the correct ISO-codes. This means the intersection of all ISO-codes in our datasets"""
    friendship_data = pd.read_csv("../data/friendship_data/countries-countries-fb-social-connectedness-index-october-2021.tsv", delimiter= "\t",keep_default_na=False)
    friendship_data = remove_self_loops(friendship_data,"user_loc","fr_loc")
    fb_data = pd.read_csv("../data/fb_data/FBCosDist.csv", delimiter= ",",keep_default_na=False)
    fb_data = remove_self_loops(fb_data,"ISO_CODE_1","ISO_CODE_2")
    fb_data, friendship_data = get_common(fb_data, friendship_data)

    all_uniques = set(fb_data["ISO_CODE_1"])
    for file in os.listdir("fb_data"):
        #We only look for csv files, not the other files
        if file.endswith(".csv"):
            #Since we already analyse the dist files, we skip them
            if file in ["FBCosDist.csv","FBEucDist.csv"]:
                continue

            file = "fb_data/" + file
            fb_data_file = pd.read_csv(file, delimiter= ",",keep_default_na = False)
            row_uniques = set(fb_data_file["ISO_CODE"])
            col_uniques = set(fb_data_file.columns)
            col_uniques.remove("ISO_CODE")
            overlap = row_uniques.intersection(all_uniques,col_uniques)
            all_uniques = overlap
    return all_uniques


def convert_df_to_dict(pandas_dataframe):
    #WE DONT USE THISis
    """It converts the dataframe into a dictionary. Here the keys are the ISO-codes and the value is the vector for training"""
    #We drop the first column, since it is just the labels along the rows
    pandas_dataframe.drop(pandas_dataframe.columns[0], axis=1,inplace = True)
    #We get the columns which we use later
    columns = pandas_dataframe.columns
    #convert the dataframe to numpy
    np_array = pandas_dataframe.to_numpy()
    #Removes the diagonal, this is to remove self loops
    a = np_array[~np.eye(np_array.shape[0],dtype=bool)].reshape(np_array.shape[0],-1)
    d = dict()
    for counter, column in enumerate(columns):
        d[column] = a[counter,:]
    return d






def load_everthing():
    """Loads in everything so the data is ready to be used for training and transforming"""
    X_list = []
    label_names, taget_data = create_target_data()
    for file in os.listdir("../data/fb_data"):
        #We only look for csv files, not the other files
        if file.endswith(".csv"):
            #idk what these files are, so we skip them for now
            if file in ["FBCosDist.csv","FBEucDist.csv"]:
                continue
            df = load("../data/fb_data/" + file,",")
            df = preprocess(df)
            df = df_to_list(df)
            X_list.append(df)
    X_dict = create_label_dict(label_names,X_list)
    Y_dict = create_label_dict(label_names, [taget_data])
    return X_dict, Y_dict

def get_feature_names(metric):
    """Function to get the list of all the feature categories i.e Facebook interest categories"""
    
    names = []
    for file in os.listdir('../data/fb_data'):
        filename = os.fsdecode(file)

        if filename in ["FBCosDist.csv","FBEucDist.csv"]: # Ignoring files with all the interest combined
                continue
        if filename.endswith(".csv"): # Ignoring all the files ending with dta as they are just copies of the csv
            if metric != None:
                if metric.title() in filename:
                    names.append(filename.strip('csv'))
            else:
                    names.append(filename.strip('csv'))

    return names # Returning names of categories

def make_total_df():

    label_names, taget_data = create_target_data()


    X_dict, Y_dict = load_everthing()

    X = list(X_dict.values())
    Y = [x[0] for x in Y_dict.values()]

    labels = np.reshape(Y,(len(Y),1))
    data = np.concatenate([X,labels],axis=1) #Adding the labels to the df of all features
    df = pd.DataFrame(data)
    feature_names = get_feature_names(None) 

    feature_names.append("Y_labels")
    df.columns = feature_names
    
    return df



def get_indv_df(metric):

    df = make_total_df()
    
    metric_li = []
    for column in df.columns:
        if metric.title() in column:
            metric_li.append(column)

    metric_df = df[metric_li]

    return metric_df