import joblib

class save_oject():
    def __init__(self, picle_file: str):
        self.gridsearch = joblib.load(picle_file)
        self.a = "hej"
    def print_att(self):
        d = self.__dict__
        for key, val in d.items():
            print(key, val)
    
    
    


def load_model():
    #load your model for further usage
    return joblib.load("test.pkl")
    print(model)

#https://stackoverflow.com/questions/34143829/sklearn-how-to-save-a-model-created-from-a-pipeline-and-gridsearchcv-using-jobli
#Scores, model, pipeline, parametre