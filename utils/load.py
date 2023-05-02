import os
import warnings
import pandas as pd
import numpy as np
from collections import defaultdict

#reflection 2 juni
#global 31 maj
#24 maj - forsvar 12-14
#bachelor aflever 15. maj
#forsvar maj 22 - juni 16

#idelt 6-8 juni

#kan ikke: 24, 30-31 maj, 1-2, 12-16 juni




def load(filename,delimiter):
    if delimiter not in [",","\t"]:
        #We only use two types of delimiters, it will raise an error if its the incorrect
        raise ValueError(f'delimiter: {delimiter} is not a correct, you should only use , or \\t')
    
    file = pd.read_csv(filename, delimiter= delimiter,keep_default_na=False)
    #Since all the files has over 100 rows and at least 3 columns, we check if it has been loaded correctly
    if (file.shape[0] < 100) and (file.shape[1] < 3):
        raise ValueError(f'The shape {file.shape} does not meet the criteria: file.shape[0] > 100 and file.shape[1] > 3')
    return file

def create_cartesian_product(list_of_iso_codes):
    """Creates a cartesian product of the ISO-codes. This is used in combination with the __ function"""
    d = dict()
    for elm in list_of_iso_codes:
        d[elm] = []
        for i in list_of_iso_codes:
            d[elm].append(elm + "_" + i)
    return pd.DataFrame(d)

def select_relevant_cells(pandas_dataframe):
    """Converts the lower half of the diagonal, INCLUDING the diagonal, into NaN values"""
    #We drop the first column, since this is our label column
    pandas_dataframe.drop(pandas_dataframe.columns[0], axis=1,inplace = True)
    #We get the dimensions
    m,n = pandas_dataframe.shape
    #We select the lower half of the diagonal, INCLUDING the diagonal, into NaN values
    pandas_dataframe[:] = np.where(np.arange(m)[:,None] >= np.arange(n),np.nan,pandas_dataframe)
    return pandas_dataframe


def create_target_data(file = "data/friendship_data/countries-countries-fb-social-connectedness-index-october-2021.tsv"):
    """Creates our taget data and a list of all the label names"""
    friendship_data = pd.read_csv(file, delimiter= "\t",keep_default_na=False)
    friendship_data = friendship_data.drop(["region"], axis=1, errors='ignore')

    friendship_data = preprocess(friendship_data)
    codes = list(friendship_data["ISO_CODE"])
    cartesian_df = create_cartesian_product(codes)
    cartesian_df.insert(0,'ISO_CODES',codes)
    cartesian_df = select_relevant_cells(cartesian_df)
    label_names = from_diagonal_to_list(cartesian_df)
    friendship_data = df_to_list(friendship_data)
    return label_names, friendship_data

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



def preprocess(pandas_dataframe):
    #We load in the relevant iso codes as a set
    with open("data/relevant_iso_codes.txt") as my_file:
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
        
        d = {k:v for k,v in d.items() if v}
        
        final_df = pd.DataFrame(d)
        
        #We sort the keys so they have the correct order
        final_df.sort_index(axis=1, inplace= True)
        iso_codes = list(headers_to_keep)
        iso_codes = list(set(iso_codes) & set(final_df.columns))

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

    #This is used for the country df's like africa and so on
    elif (pandas_dataframe.shape[0] == pandas_dataframe.shape[1] - 1) and (pandas_dataframe.shape[0] < 65):
        final_df = pandas_dataframe 

    else:
        print(pandas_dataframe.shape)
        warnings.warn(f"The shape of the dataframe is likely to be wrong. Expected df.shape[1] == 3 or 226, but got {pandas_dataframe.shape}")
    return final_df



def df_to_list(processed_dataframe):
    """Takes the preprocessed dataframe and converts it into a list"""
    processed_dataframe = select_relevant_cells(processed_dataframe)
    processed_dataframe = from_diagonal_to_list(processed_dataframe)
    return processed_dataframe



def create_label_dict(iso_codes: list, x_values: list ):
    """takes two lists, one containing all ISO-codes and another being a nested list of our X-values"""
    new_list = list(zip(*x_values))
    label_dict = dict()
    for name, value in zip(iso_codes, new_list):
        label_dict[name] = value
    return label_dict




def get_feature_names(metric):
    """Function to get the list of all the feature categories i.e Facebook interest categories"""
    
    names = []
    for file in sorted(os.listdir('data/fb_data')):
        filename = os.fsdecode(file)

        if filename in ["FBCosDist.csv","FBEucDist.csv"]: # Ignoring files with all the interest combined
                continue
        if filename.endswith(".csv"): # Ignoring all the files ending with dta as they are just copies of the csv
            if metric != None: # If we just want to look at all metrics
                if metric.lower() in filename.lower(): # Seeing if the metric is in any of the filenames
                    names.append(filename.strip('csv'))
            else:
                names.append(filename.strip('csv')) 
    return names # Returning names of categories within the distance

def load_everthing_with_distance(test_size = 0, val_size = 0,transform_function = None):
    """You can pass it a function to transform the value. 
    An example would be transform_function = np.log10 """
    x,y = load_everthing()
    dist = pd.read_csv("data/misc/iso_dist.csv")
    dist_dict = dict()
    for _, row in dist.iterrows():
        dist_dict[row.iso_o] = row.dist
    iso_to_keep = set(dist["iso_o"])
    newdict = {k: y[k] for k in iso_to_keep}
    new_y = dict(sorted(newdict.items()))
    
    new_x = defaultdict(dict)
    for dist_metric, val in x.items():
        for iso_code, feature in val.items():
            if iso_code in iso_to_keep:
                feature = list(feature)

                #This value we append here is the dist feature. We can transform it here
                if transform_function != None:
                    feature.append(float(transform_function(dist_dict[iso_code])))
                else:
                    feature.append(float(dist_dict[iso_code]))
                new_x[dist_metric][iso_code] = feature
    
    if (test_size != 0) or (val_size != 0):
        new_x, new_y = test_train_val_split(new_x,new_y,test_size = test_size,val_size = val_size)

    return new_x, new_y
        


import random
def make_split_dict(list_of_keys, test_size, val_size, seed = 1234):
    """Takes a list of keys, and the size of the test and val dataset as percentages"""
    random.seed(seed)
    d = dict()
    total = len(list_of_keys)
    all_keys = list_of_keys
    test = int(total*test_size)
    val = int(total*val_size)
    test_keys = random.sample(all_keys, test)
    all_keys = [i for i in all_keys if i not in test_keys]
    val_keys = random.sample(all_keys, val)
    all_keys = [i for i in all_keys if i not in val_keys]
    d["train"] = all_keys
    d["test"] = test_keys
    d["val"] = val_keys
    return d



def test_train_val_split(x,y,test_size, val_size):
    new_x = defaultdict(lambda: defaultdict(dict))
    #We just sample from CosDist. The reason is that we see no reason to sample independently from the different
    #distance metrics, since the goal is simply to split the data up and compare the different distance metrics equally.
    d = make_split_dict(list(x["CosDist"].keys()), test_size, val_size)
    for distance_metric, x_dict in x.items():
        for iso, feature in x_dict.items():
            if iso in d["train"]:
                new_x["train"][distance_metric][iso] = feature
            elif iso in d["test"]:
                new_x["test"][distance_metric][iso] = feature
            elif iso in d["val"]:
                new_x["val"][distance_metric][iso] = feature
    new_y = defaultdict(dict)
    for iso, label in y.items():
        if iso in d["train"]:
            new_y["train"][iso] = label
        if iso in d["test"]:
            new_y["test"][iso] = label
        if iso in d["val"]:
            new_y["val"][iso] = label
    return new_x, new_y

def test_train_val_split_continents(x,y,test_size, val_size):
    def rec_dd():
        return defaultdict(rec_dd)


    #return new_x, new_y
    new_x = rec_dd()
    d = rec_dd()
    #We just sample from CosDist. The reason is that we see no reason to sample independently from the different
    #distance metrics, since the goal is simply to split the data up and compare the different distance metrics equally.
    test_size = 0.2
    val_size = 0
    for distance_metric, idk in x.items():
        for continent, x_dict in idk.items():
            d[continent] = make_split_dict(list(x["CosDist"][continent].keys()), test_size, val_size)
            for iso, feature in x_dict.items():
                if iso in d[continent]["train"]:
                    new_x["train"][distance_metric][continent][iso] = feature
                elif iso in d[continent]["test"]:
                    new_x["test"][distance_metric][continent][iso] = feature
                elif iso in d[continent]["val"]:
                    new_x["val"][distance_metric][continent][iso] = feature
    new_y = rec_dd()
    for continent, val in y.items():
        for iso, label in val.items():
            if iso in d[continent]["train"]:
                new_y["train"][continent][iso] = label
            if iso in d[continent]["test"]:
                new_y["test"][continent][iso] = label
            if iso in d[continent]["val"]:
                new_y["val"][continent][iso] = label
    return new_x, new_y


def load_everthing(test_size = 0, val_size = 0):
    """Loads in everything so the data is ready to be used for training and transforming"""
    X_dict = dict()
    Y_dict = dict()
    d = dict()
    d["CosDist"] = []
    d["EucDist"] = []
    d["HetDist"] = []
    d["ManDist"] = []
    label_names, taget_data = create_target_data() 

    for file in (os.listdir("data/fb_data")):
        #We only look for csv files, not the other files
        if file.endswith(".csv"):
            #idk what these files are, so we skip them for now
            if file in ["FBCosDist.csv","FBEucDist.csv"]:
                continue

            df = load("data/fb_data/" + file,",")
            df = preprocess(df)
            df = df_to_list(df)
            if "CosDist" in file:
                d["CosDist"].append(df)
            elif "EucDist" in file:
                d["EucDist"].append(df)
            elif "HetDist" in file:
                d["HetDist"].append(df)
            elif "ManDist" in file:
                d["ManDist"].append(df)
            
    for key,val in d.items():
        X_dict[key] = create_label_dict(label_names,val)
    
    Y_dict = create_label_dict(label_names, [taget_data])
    if (test_size != 0) or (val_size != 0):
        X_dict, Y_dict = test_train_val_split(X_dict,Y_dict,test_size,val_size) 

    return X_dict, Y_dict


def load_all_distance_metrics(test_size = 0, val_size = 0, with_distance = False):
    if with_distance:
        x,y = load_everthing_with_distance(test_size= test_size, val_size= val_size)
    else:
        x,y = load_everthing(test_size= test_size, val_size= val_size)
    new_dict = {}
    for row in x:
        new_dict[row] = {}
        for distance_metric in x[row]:
            for iso_code in x[row][distance_metric]:
                if iso_code not in new_dict[row]:
                    new_dict[row][iso_code] = []
                new_dict[row][iso_code].extend(x[row][distance_metric][iso_code])
    return new_dict, y


def sort_by_country(file_name):
    if "europe" in file_name:
        return "europe"
    elif "africa" in file_name:
        return "africa"
    elif "asia" in file_name:
        return "asia"
    elif "oceania" in file_name:
        return "oceania"
    elif "americas" in file_name:
        return "americas"

def load_everthing_with_continents(test_size = 0, val_size = 0):
    """Loads in everything so the data is ready to be used for training and transforming"""

    Y_dict = defaultdict(dict)
    d = dict()
    d["CosDist"] = defaultdict(list)
    d["EucDist"] = defaultdict(list)
    d["HetDist"] = defaultdict(list)
    d["ManDist"] = defaultdict(list)
    
    label_d = defaultdict(dict)
    label_dir = "data/friendship_data_continents_new/"
    for file in os.listdir(label_dir):
        if file.endswith("tsv"):
            label_names, target_data = create_target_data(label_dir + file)
            country = file.replace(".tsv","").lower()
            label_d[country]["label_names"] = label_names
            label_d[country]["target_data"] = target_data

    label_names, target_data = create_target_data() 
    path = "fb_data_continents_new/"
    for file in os.listdir(f"data/{path}"):
        #We only look for csv files, not the other files
        if file.endswith(".csv"):
            #idk what these files are, so we skip them for now
            if file in ["FBCosDist.csv","FBEucDist.csv"]:
                continue
            
            df = load(f"data/{path}" + file,",")
            df = preprocess(df)
            df = df_to_list(df)
            if "CosDist" in file:
                d["CosDist"][sort_by_country(file)].append(df)
            elif "EucDist" in file:
                d["EucDist"][sort_by_country(file)].append(df)
            elif "HetDist" in file:
                d["HetDist"][sort_by_country(file)].append(df)
            elif "ManDist" in file:
                d["ManDist"][sort_by_country(file)].append(df)


    for country, val in label_d.items():
        Y_dict[country] = create_label_dict(val["label_names"], [val["target_data"]])

    
    final = defaultdict(dict)
    for distance, country_dict in d.items():
        for country, val in country_dict.items():
            label_names = [x for x in Y_dict[country].keys()]
            final[distance][country] = create_label_dict(label_names, val)

    #We need to sort the continents
    new_x = defaultdict(dict)
    for distance_metric, x_dict in final.items():
        new_x[distance_metric] = dict( sorted(x_dict.items(), key=lambda a: a[0].lower()) )
    new_y = dict(sorted(Y_dict.items(), key=lambda x: x[0].lower()) )

    if (test_size != 0) or (val_size != 0):
        new_x, new_y = test_train_val_split(new_x,new_y,test_size,val_size) 
    #x,y = test_train_val_split_continents(new_x, new_y, test_size= 0.2, val_size= 0)
    return new_x, new_y
    




def load_everthing_old():
    """Loads in everything so the data is ready to be used for training and transforming"""
    X_list = []
    label_names, taget_data = create_target_data()
    for file in sorted(os.listdir("data/fb_data")):
        #We only look for csv files, not the other files
        if file.endswith(".csv"):
            #idk what these files are, so we skip them for now
            if file in ["FBCosDist.csv","FBEucDist.csv"]:
                continue
            df = load("data/fb_data/" + file,",")
            df = preprocess(df)
            df = df_to_list(df)
            X_list.append(df)
    X_dict = create_label_dict(label_names,X_list)
    Y_dict = create_label_dict(label_names, [taget_data])
    return X_dict, Y_dict


def make_total_df():
    """ This function returns a dataframe with both the X-values and the Y-values along 
    with the country labels as idices e.g DK-SE"""

   # label_names, taget_data = create_target_data()

    X_dict, Y_dict = load_everthing_old()

    X = list(X_dict.values())
    Y = [x[0] for x in Y_dict.values()]
    label_names = list(X_dict.keys())


    labels = np.reshape(Y,(len(Y),1))
    data = np.concatenate([X,labels],axis=1) #Adding the labels to the df of all features
    
    df = pd.DataFrame(data)
 
    feature_names = get_feature_names(None) 
    feature_names.append("Y_labels") # Adding the actual labels
    df.columns = feature_names

    df['c_to_c'] = label_names # Adding th ecountry to country column
    df = df.set_index('c_to_c') # Using the country to country as the index
 
    return df



def get_indv_df(metric):

    """Returns the same data frame as above but with only the 
    cos, euc, man, het values """

    df = make_total_df()
    
    metric_li = []
    for column in df.columns:
        if metric.title() in column:
            metric_li.append(column)

    metric_df = df[metric_li]

    return metric_df



def load_iso_dict():
    """Loads in a dictionary of iso codes"""
    d = dict()
    for file in os.listdir("data/iso_codes"):
        with open(f"data/iso_codes/{file}") as f:
            f = f.readlines()
            d[file.replace(".txt","")] = [x.replace("\n","") for x in f]
    return d