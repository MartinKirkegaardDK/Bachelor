import pandas as pd
import numpy as np

def check_non_positive_values(metric,category):
    file_path = f"data/fb_data/FB{metric}Dist_{category}.csv"
    df = pd.read_csv(file_path, index_col=0)
    non_positive_values = df[df <= 0].dropna()
    if not non_positive_values.empty:
        print(f"The following non-positive values were found for {metric}:\n{non_positive_values}")
    else:
        print(f"No non-positive values were found for {metric}.")

check_non_positive_values("Cos","Education")
check_non_positive_values("Euc","Education")
check_non_positive_values("Man","Education")
check_non_positive_values("Het","Education")