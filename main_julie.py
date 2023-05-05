#from src.examples.random_forrest import model
from src.drivers.plots.hist_compare_r2_scores import rf, train_lr_only_dist,lf

rf()
print("Done with rf")

print("Training lr")
train_lr_only_dist()
print("Done Training lr")

lf()
print("Done making lr plot")
