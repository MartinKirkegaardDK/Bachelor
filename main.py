
from src.PCA.pca import pca_func
from utils.lib import make_total_df

df = make_total_df()

print(pca_func(df))