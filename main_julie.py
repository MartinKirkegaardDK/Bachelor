from src.examples.random_forrest import model

from src.PCA.pcr import scree_plot_all_data, scree_plot_for_indv_metrics, rf_with_distance, rf_without_distance
import numpy as np
import matplotlib as plt

scree_plot_all_data()
scree_plot_for_indv_metrics()
rf_with_distance()
rf_without_distance()