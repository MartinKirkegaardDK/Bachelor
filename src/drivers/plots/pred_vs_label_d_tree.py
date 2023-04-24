from utils.utilities import get_pred_and_labels
from utils.plots import plot_r2, plot_gpt4
from joblib import load

def run():
    clf = load("models_to_save/.pkl")
    pred, labels = get_pred_and_labels(clf,n = 0, with_distance= False)
    plot_r2(clf, pred, labels,"Predicted_vs_labels_without_distance")