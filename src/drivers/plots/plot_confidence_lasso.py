from utils.utilities import bootstrap, gen_feature_dict_lasso
from utils.plots import plot_confidence_interval
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from utils.utilities import bootstrap_continents

n = 10

def confidence_plot_lasso_without_dist():
    pipe = Pipeline([("StandardScaler",StandardScaler()),("Lasso_regressor",LassoCV(max_iter= 10000))])
    s = bootstrap(pipeline= pipe, param_grid= {}, n = n)
    feature_dict = gen_feature_dict_lasso(s)
    plot_confidence_interval(feature_dict,"Coefficient_estimate_lasso_without_distance")

def confidence_plot_lasso_with_dist():
    pipe = Pipeline([("StandardScaler",StandardScaler()),("Lasso_regressor",LassoCV(max_iter= 10000))])
    s = bootstrap(pipeline= pipe, param_grid= {},with_dist= True, n = n)
    feature_dict = gen_feature_dict_lasso(s, with_dist= True)
    plot_confidence_interval(feature_dict,"Coefficient_estimate_lasso_with_distance")


def confidence_plot_lasso_continents():
    pipe = Pipeline([("StandardScaler",StandardScaler()),("Lasso_regressor",LassoCV(max_iter= 10000))])
    s = bootstrap_continents(pipeline= pipe, param_grid= {},n = n)
    for continent, value in s.items():

        feature_dict = gen_feature_dict_lasso(value)
        plot_confidence_interval(feature_dict,f"Coefficient_estimate_lasso",continent= continent)

def run():
    print("plotting confidence_plot_lasso_without_dist")
    confidence_plot_lasso_without_dist()
    print("plotting confidence_plot_lasso_with_dist")
    confidence_plot_lasso_with_dist()
    print("plotting confidence_plot_lasso_continents")
    confidence_plot_lasso_continents()

