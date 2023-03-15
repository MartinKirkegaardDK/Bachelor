from src.examples.gridsearch import run
import pandas as pd
#run()

from utils.save_object import save_oject

from utils.utilities import naming

model_object = save_oject("models/test_1.pkl")

#model_object.print_att()
d = pd.DataFrame(model_object.gridsearch.cv_results_)
print(d)
print(d.columns)

#print(dir(model))
#best_model = model.best_estimator_
#best_model.predict()


#print(naming("runs"))