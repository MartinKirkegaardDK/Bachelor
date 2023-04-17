from utils.utilities import bootstrap, gen_feature_dict
from utils.plots import plot_confidence_interval

from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def run():
    pipe = Pipeline([("StandardScaler",StandardScaler()),("Lasso_regressor",LassoCV(max_iter= 10000))])
    s = bootstrap(pipeline= pipe, param_grid= {})
    feature_dict = gen_feature_dict(s)
    plot_confidence_interval(feature_dict)