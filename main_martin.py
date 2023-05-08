from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from utils.utilities import process, run_confidence_interval, file_name
from utils.gridsearch import gridsearch_new
from utils.load import loader
from utils.ml_models import lasso, lasso_pca, rf, rf_pca, ridge
from utils.plots import plot_r2
import warnings
warnings.filterwarnings('ignore')



def do_everything(with_distance, with_all_dist_metrics, model):
    pipeline, param_grid = model()
    #print(pipeline.steps)
    x,y = loader(with_distance, with_all_dist_metrics)
    x_train, y_train, x_test, y_test = process(x,y)
    #If we use all the dist metrics, then we dont loop and simply run gridsearch and so on

    if with_all_dist_metrics:
        name = file_name(with_distance, with_all_dist_metrics, pipeline)

        distance_metric = "all"
        obj = gridsearch_new(x_train, y_train, pipeline, param_grid, distance_metric, with_distance)
        search = obj.gridsearch
        if "PCA" not in name:
            pass
            run_confidence_interval(pipeline, param_grid, distance_metric, with_distance,with_all_dist_metrics,f"Coefficient estimate {name}")
        pred = search.best_estimator_.predict(x_test)
        plot_r2(pred, y_test,f"Predicted vs labels {name}")
    else:
        for x_train_elm, x_test_elm in zip(x_train.items(), x_test.items()):

            distance_metric = x_train_elm[0]
            name = file_name(with_distance, with_all_dist_metrics, pipeline,distance_metric)


            x_train_elm = x_train_elm[1]
            x_test_elm = x_test_elm[1]
            obj = gridsearch_new(x_train_elm, y_train, pipeline, param_grid, distance_metric, with_distance)
            search = obj.gridsearch
            run_confidence_interval(pipeline, param_grid, distance_metric, with_distance,with_all_dist_metrics,f"Coefficient estimate {name}")
            pred = search.best_estimator_.predict(x_test_elm)
            plot_r2(pred, y_test,f"Predicted vs labels {name}")

def run_all():
    with_all_dist_metrics = [True,False] 
    with_distances = [True, False]
    models = [lasso, lasso_pca, rf, rf_pca]
    for all_dist_metric in with_all_dist_metrics:
        for with_distance in with_distances:
            for model in models:
                print("with_distance: ",with_distance) 
                print("all_dist_metric: ",all_dist_metric)
                print("model: ",model)
                do_everything(with_distance, all_dist_metric, model)
                print("*"*75)


def run1():
    with_all_dist_metrics = [True] 
    with_distances = [True]
    models = [lasso, lasso_pca, rf, rf_pca]
    for all_dist_metric in with_all_dist_metrics:
        for with_distance in with_distances:
            for model in models:
                print("with_distance: ",with_distance) 
                print("all_dist_metric: ",all_dist_metric)
                print("model: ",model)
                do_everything(with_distance, all_dist_metric, model)
                print("*"*75)

def run2():
    with_all_dist_metrics = [False] 
    with_distances = [False]
    models = [lasso, lasso_pca, rf, rf_pca]
    for all_dist_metric in with_all_dist_metrics:
        for with_distance in with_distances:
            for model in models:
                print("with_distance: ",with_distance) 
                print("all_dist_metric: ",all_dist_metric)
                print("model: ",model)
                do_everything(with_distance, all_dist_metric, model)
                print("*"*75)

def run3():
    with_all_dist_metrics = [True] 
    with_distances = [False]
    models = [lasso, lasso_pca, rf, rf_pca]
    for all_dist_metric in with_all_dist_metrics:
        for with_distance in with_distances:
            for model in models:
                print("with_distance: ",with_distance) 
                print("all_dist_metric: ",all_dist_metric)
                print("model: ",model)
                do_everything(with_distance, all_dist_metric, model)
                print("*"*75)

def run4():
    with_all_dist_metrics = [False] 
    with_distances = [True]
    models = [lasso, lasso_pca, rf, rf_pca]
    for all_dist_metric in with_all_dist_metrics:
        for with_distance in with_distances:
            for model in models:
                print("with_distance: ",with_distance) 
                print("all_dist_metric: ",all_dist_metric)
                print("model: ",model)
                do_everything(with_distance, all_dist_metric, model)
                print("*"*75)

def run5():
    with_all_dist_metrics = [True,False] 
    with_distances = [True, False]
    models = [ridge]
    for all_dist_metric in with_all_dist_metrics:
        for with_distance in with_distances:
            for model in models:
                print("with_distance: ",with_distance) 
                print("all_dist_metric: ",all_dist_metric)
                print("model: ",model)
                do_everything(with_distance, all_dist_metric, model)
                print("*"*75)

def test():
    run5()

if __name__ == '__main__': 
    test()