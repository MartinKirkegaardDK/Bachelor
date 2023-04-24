from src.examples.random_forrest import model

from src.PCA.pca import scree_plot, myplot, pca_func
import numpy as np
import matplotlib as plt
x, pca, pcamodel = pca_func()
#myplot(pca[:,0:2],np.transpose(pcamodel.components_[0:2, :]),list(x.columns))

scree_plot()
