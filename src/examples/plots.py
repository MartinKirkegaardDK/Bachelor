from utils.utilities import get_pred_and_labels
from utils.plots import plot_r2, plot_gpt4
from joblib import load

def run():
    clf = load("models_to_save/linear_reg_lasso.pkl")
    pred, labels = get_pred_and_labels(clf,n = 0)
    plot_r2(clf, pred, labels)
    plot_gpt4(pred,labels)