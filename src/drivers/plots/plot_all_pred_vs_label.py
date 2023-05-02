import os
from joblib import load
from utils.utilities import get_pred_and_labels
from utils.plots import plot_r2

def run():
    for file in os.listdir("models_to_save"):
        if file.endswith(".pkl"):
            clf = load(f"models_to_save/{file}")
            print(file)
            metrics = False
            if "all_distance_metrics" in file:
                metrics = True
            
            if "with_dist" in file:
                pred, labels = get_pred_and_labels(clf,n = 0, with_distance= True,all_distance_metrics= metrics)
            else:
                pred, labels = get_pred_and_labels(clf,n = 0, with_distance= False, all_distance_metrics= metrics)
            
            name = file.replace(".pkl","")
            name = name.replace("_"," ")
            plot_r2(pred, labels,f"Predicted vs labels {name}")
            #break
        