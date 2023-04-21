from utils.utilities import bootstrap, gen_feature_dict_lasso
from utils.plots import plot_confidence_interval

from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def run():
    pipe = Pipeline([("StandardScaler",StandardScaler()),("Lasso_regressor",LassoCV(max_iter= 10000))])
    s = bootstrap(pipeline= pipe, param_grid= {},with_dist= True, n = 100)
    feature_dict = gen_feature_dict_lasso(s, with_dist= True)
    plot_confidence_interval(feature_dict,"Coefficient_estimate_lasso_with_distance")