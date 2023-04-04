import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

def plot_gpt4(pred, labels):
    """the gpt4 plot, idk"""
    d = dict()
    d["pred"] = list(pred)
    d["labels"] = labels

    df = pd.DataFrame(d)
    df.sort_values("labels",inplace = True)
    
    t = list(range(len(pred)))
    plt.scatter(t, df["pred"], label = "pred")
    plt.scatter(t,df["labels"], label = "labels")
    plt.xlabel("Index sorted by label")
    plt.ylabel("Values")
    plt.title("Vores awesome plot som gpt-4 har hjulpet med :)")
    plt.legend()
    plt.savefig("plots/idk.png")
    plt.show()

def plot_r2(clf, pred, labels):
    """This is the scatterplot with the r2 line"""
    plt.figure(figsize=(5,5))
    plt.scatter(labels, pred, c='crimson')
    plt.yscale('log')
    plt.xscale('log')
    p1 = max(max(pred), max(labels))
    p2 = min(min(pred), min(labels))
    plt.plot([p1, p2], [p1, p2], 'b-',label = f"$R^2$ score = {round(clf.cv_results_['mean_test_score'][0],2)}")
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')
    plt.legend()
    plt.savefig("plots/pred_vs_labels.png")
    plt.show()

def plot_confidence_interval(feature_dict):
    data_dict = defaultdict(list)
    for key, val in feature_dict.items():
        data_dict['category'].append(key)
        data_dict['lower'].append(min(val))
        data_dict['upper'].append(max(val))
    dataset = pd.DataFrame(data_dict)
    for lower,upper,y in zip(dataset['lower'],dataset['upper'],range(len(dataset))):
        plt.plot((lower,upper),(y,y),'ro-',color='orange')
    plt.title("Coefficient estimate within 0.95 confidence interval")
    plt.xlabel("Coefficient estimate")
    plt.yticks(range(len(dataset)),list(dataset['category']))
    plt.subplots_adjust(left = 0.25)
    plt.savefig("plots/Coefficient_estimate.png")
    plt.show()