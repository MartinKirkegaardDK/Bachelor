from src.examples.random_forrest import model

from src.PCA.pcr import scree_plot_all_data, scree_plot_for_indv_metrics, rf_with_distance, rf_without_distance
import numpy as np
import matplotlib as plt
from src.PCA.pca import myplot, pca_func
from utils.load import get_indv_df

cos_df = get_indv_df('cos')

x, pca, pcamodel = pca_func()


names = []
for i in x.columns:
    discard, cat = i.split("_")
    cat = cat.strip(".")

    if cat in "NonLocalBus":
        cat = "NonLocalBusiness"
    names.append(cat)




myplot(pca[:,0:2],np.transpose(pcamodel.components_[0:2, :]),names)
print(names)
