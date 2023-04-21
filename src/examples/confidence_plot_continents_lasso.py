from utils.utilities import gen_feature_dict_lasso
from utils.plots import plot_confidence_interval
from utils.utilities import bootstrap_continents

from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def run():
    pipe = Pipeline([("StandardScaler",StandardScaler()),("Lasso_regressor",LassoCV(max_iter= 10000))])
    s = bootstrap_continents(pipeline= pipe, param_grid= {})
    for continent, value in s.items():
        #t = [x.gridsearch for x in value]
        feature_dict = gen_feature_dict_lasso(value)
        plot_confidence_interval(feature_dict,f"Coefficient_estimate_lasso_{continent}",continent= continent)