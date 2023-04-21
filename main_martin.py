from src.examples.confidence_plot_continents_lasso import run
#run()
import pandas as pd
from utils.load import load_everthing
dist = pd.read_csv("data/misc/iso_dist.csv")

x,y = load_everthing()

for dist_metric, val in x.items():
    print(val)