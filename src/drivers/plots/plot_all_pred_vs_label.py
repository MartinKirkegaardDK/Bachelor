import os
from joblib import load
from utils.utilities import get_pred_and_labels
from utils.plots import plot_r2

for file in os.listdir("models_to_save"):
    if file.endswith(".pkl"):
        clf = load(f"models_to_save/{file}")
        if "with_" in file:
            pred, labels = get_pred_and_labels(clf,n = 0, with_distance= True)
        else:
            pred, labels = get_pred_and_labels(clf,n = 0, with_distance= False)
        name = file.replace(".pkl","")
        plot_r2(clf, pred, labels,f"Pred_vs_labels_{name}")