from utils.utilities import process
from utils.load import loader
from utils.gridsearch import gridsearch_new
from utils.ml_models import ridge


def train_model(with_distance, with_all_dist_metrics, model,distance_metric = None):
    pipeline, param_grid = model()
    #print(pipeline.steps)
    x,y = loader(with_distance, with_all_dist_metrics)
    x_train, y_train, x_test, y_test = process(x,y)
    #If we use all the dist metrics, then we dont loop and simply run gridsearch and so on

    if with_all_dist_metrics:
        #name = file_name(with_distance, with_all_dist_metrics, pipeline)

        distance_metric = "all"
        obj = gridsearch_new(x_train, y_train, pipeline, param_grid, distance_metric, with_distance)
        search = obj.gridsearch.best_estimator_
    else:
        if distance_metric  not in ["CosDist","EucDist","HetDist","ManDist"]:
            raise ValueError('distance_metric has to be in ["CosDist","EucDist","HetDist","ManDist"]') 
        x_train = x_train[distance_metric]
        distance_metric = distance_metric
        obj = gridsearch_new(x_train, y_train, pipeline, param_grid, distance_metric, with_distance)
        search = obj.gridsearch.best_estimator_

    return search, x_test, y_test

#Sådan kører du modelen. Du importerer en model fra utils.ml_models og så giver du den parametre.
#Hvis du vil træne på alle distance metrics, så behøver du ikke give den en distance_metric
#from utils.ml_models import ridge
#from utils.ml_model import rf
#model, x_test, y_test = train_model(True, False, ridge, distance_metric= "CosDist")
#model, x_test, y_test = train_model(True, True, rf)