#from utils.load import make_total_df
#from src.PCA.correlation import top_abs_correlations
#from src.Visualisations.dist_labels import logged_all
#from src.Visualisations.dist_labels import logged_all
#from src.PCA.non_pos import check_non_positive_values
#from src.PCA.check_links import check_link_values
#from src.PCA.correlation import top_abs_correlations
#from src.Visualisations.Continents.network import calculate_betweenness_centrality, draw_graph
from src.Visualisations.Continents.continents_fb import split_by_continent

#from src.Visualisations.Continents.continents_friend import split_social_continent
#from src.Visualisations.distributions import std_dist_plots
#print(dist_labels("data/friendship_data/countries-countries-fb-social-connectedness-index-october-2021.tsv","DK",15))

#print(top_abs_correlations(make_total_df(),10))


from src.drivers.plots.plot_confidence_lasso import run as plot_confidence_lasso
from src.drivers.plots.plot_confidence_lasso_all_dist_metrics import run as plot_confidence_lasso_all_dist_metrics
from src.drivers.plots.plot_confidence_rf import run as plot_confidence_rf
from src.drivers.plots.plot_confidence_rf_all_dist_metrics import run as plot_confidence_rf_all_dist_metrics

print("running plot_confidence_lasso")
plot_confidence_lasso()
print("plot_confidence_lasso_all_dist_metrics")
plot_confidence_lasso_all_dist_metrics()
print("plot_confidence_rf")
plot_confidence_rf()
print("plot_confidence_rf_all_dist_metrics")
plot_confidence_rf_all_dist_metrics()