

import sys
sys.path.append('..')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lib

def correlation_matrix():
    # Getting the data 
    df4 = lib.make_total_df()

    # Dropping the column Y_labels
    df4 = df4.drop(columns=['Y_labels'])

    # Sorting the values and printing the correlations + dropping duplicates 
    print("Top correlations:",df4.corr().unstack().sort_values().drop_duplicates())

    # Plotting the correlation matrix
    corr_plt = df4.corr().style.background_gradient(cmap='RdBu_r', axis=None).set_precision(2) 
    # By default it uses the Pearson correlation   

    return corr_plt
    
correlation_matrix()

# Getting the top correlations 
def redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop


def top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top 10 correlations:\n",top_abs_correlations(lib.make_total_df(),10))