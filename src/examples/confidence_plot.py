from utils.utilities import bootstrap, gen_feature_dict
from utils.plots import plot_confidence_interval

def run():
    s = bootstrap()
    feature_dict = gen_feature_dict(s)
    plot_confidence_interval(feature_dict)