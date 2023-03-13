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

def load(filename,delimiter):
    if delimiter not in [",","\t"]:
        #We only use two types of delimiters, it will raise an error if its the incorrect
        raise ValueError(f'delimiter: {delimiter} is not a correct, you should only use , or \\t')
    
    file = pd.read_csv(filename, delimiter= delimiter,keep_default_na=False)
    #Since all the files has over 100 rows and at least 3 columns, we check if it has been loaded correctly
    if (file.shape[0] < 100) and (file.shape[1] < 3):
        raise ValueError(f'The shape {file.shape} does not meet the criteria: file.shape[0] > 100 and file.shape[1] > 3')
    return file

def preprocess(pandas_dataframe):
    #We load in the relevant iso codes as a set
    with open("../data/relevant_iso_codes.txt") as my_file:
        x = my_file.read()
        headers_to_keep = set(x.split("\n"))

    #There are two types of files, those with 3 columns and those with more.
    #We transform the 3 columns into the one with 226 columns
    if pandas_dataframe.shape[1] == 3:
        #We only keep the correct iso codes
        pandas_dataframe = pandas_dataframe[(pandas_dataframe.iloc[:,0].isin(headers_to_keep)) & (pandas_dataframe.iloc[:,1].isin(headers_to_keep))]
        
        #We transform the data into a dictionary style, this is done so we can transform it into a pandas dataframe
        d = dict()
        for val in headers_to_keep:
            subset = pandas_dataframe[pandas_dataframe.iloc[:,0] == val]
            d[val] = list(subset.iloc[:,-1])
        

        final_df = pd.DataFrame(d)
        #We sort the keys so they have the correct order
        final_df.sort_index(axis=1, inplace= True)
        iso_codes = list(headers_to_keep)
        iso_codes.sort()
        #We insert the last column to transform it into a matrix
        final_df.insert(0,'ISO_CODE',iso_codes)

    elif pandas_dataframe.shape[1] == 226:
        #We only keep the correct headers
        pandas_dataframe = pandas_dataframe[pandas_dataframe["ISO_CODE"].isin(headers_to_keep)]
        #We add a column with the iso codes so that it is a matrix instead of a normal dataframe
        headers_to_keep.add("ISO_CODE")
        final_df =  pandas_dataframe[pandas_dataframe.columns.intersection(headers_to_keep)]
    #Since we only expect shape[1] == 226 or 3, we raise a warning, but not an error
    else:
        warnings.warn(f"The shape of the dataframe is likely to be wrong. Expected df.shape[1] == 3 or 226, but got {pandas_dataframe.shape}")
    return final_df

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

def from_diagonal_to_list(pandas_dataframe):
    """Takes a dataframe with the lower half of the diagonal being nan values and converts it into a single list"""
    li = []
    for column in pandas_dataframe:
        temp = list(pandas_dataframe[column])
        for cell in temp:
            #If the cell is not nan, then we append it to the list
            if pd.isnull(cell) == False:
                li.append(cell)
    return li

def select_relevant_cells(pandas_dataframe):
    """Converts the lower half of the diagonal, INCLUDING the diagonal, into NaN values"""
    #We drop the first column, since this is our label column
    pandas_dataframe.drop(pandas_dataframe.columns[0], axis=1,inplace = True)
    #We get the dimensions
    m,n = pandas_dataframe.shape
    #We select the lower half of the diagonal, INCLUDING the diagonal, into NaN values
    pandas_dataframe[:] = np.where(np.arange(m)[:,None] >= np.arange(n),np.nan,pandas_dataframe)
    return pandas_dataframe

def df_to_list(processed_dataframe):
    """Takes the preprocessed dataframe and converts it into a list"""
    processed_dataframe = select_relevant_cells(processed_dataframe)
    processed_dataframe = from_diagonal_to_list(processed_dataframe)
    return processed_dataframe

def create_cartesian_product(list_of_iso_codes):
    """Creates a cartesian product of the ISO-codes. This is used in combination with the __ function"""
    d = dict()
    for elm in list_of_iso_codes:
        d[elm] = []
        for i in list_of_iso_codes:
            d[elm].append(elm + "_" + i)
    return pd.DataFrame(d)

def create_label_dict(iso_codes: list, x_values: list ):
    """takes two lists, one containing all ISO-codes and another being a nested list of our X-values"""
    new_list = list(zip(*x_values))
    label_dict = dict()
    for name, value in zip(iso_codes, new_list):
        label_dict[name] = value
    return label_dict

def create_target_data():
    """Creates our taget data and a list of all the label names"""
    friendship_data = pd.read_csv("../data/friendship_data/countries-countries-fb-social-connectedness-index-october-2021.tsv", delimiter= "\t",keep_default_na=False)
    friendship_data = preprocess(friendship_data)
    codes = list(friendship_data["ISO_CODE"])
    cartesian_df = create_cartesian_product(codes)
    cartesian_df.insert(0,'ISO_CODES',codes)
    cartesian_df = select_relevant_cells(cartesian_df)
    label_names = from_diagonal_to_list(cartesian_df)
    friendship_data = df_to_list(friendship_data)
    return label_names, friendship_data


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