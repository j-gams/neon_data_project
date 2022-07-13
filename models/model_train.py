### train model, report info

### import ...
import numpy as np

def train(self, dataset, modelclass, hparams, save_names, params):
    k_folds = params["folds"]
    
    #logger = 
    #performance = []
    #models = []

    for i in range(k_folds):
        model = modelclass(hparams)
        model.train(dataset.train[i], dataset.validation[i], tepochs=hparams["epochs"])
        yhat = model.predict(datase)
