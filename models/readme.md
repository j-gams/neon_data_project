# MODEL TRAINING AND EVALUATION PIPELINE
## Overview
The general structure of the training/evaluation pipeline is as follows:
#### Data Management
Two files in the create_data directory are used to manage data for the training and evaluation process.
These interact with the directory and file structure of the dataset to greatly streamline loading and using the dataset.

dat_obj.py contains a dataset class. Initializing this with a dataset will load all the splits and folds and will store both the raw metadata csv and a data fold object for each. This object also provides various tools for working with the dataset as a whole, like removing samples based on thresholds such as distance from the nearest GEDI centroid or dominant land cover value.

datacube_set.py contains a data fold class which extends keras.utils.Sequence. This keeps track of metadata, y values, and X datacube references. Depending on the mem_sensitive parameter, This may also load the entire split or validation fold into memory. In general, though, this object is designed to facilitate batch loading of the data. It allows for batch computing of the fold means and standard deviations, loading csv or h5 data, shuffling on epoch end for tensorflow models, and filtering out specified channels of the datacube. It will also provide a vectorized version of the data that can be used in non-convolutional models. This is complex and delicate.

#### Training Framework
Two files in the models directory manage the training and evaluation of models in a standardized way.

train_frame.py loads the dataset and initializes model hyperparameters and traininng parameters. Any changes to model parameters should be made here. It then sends the model to model_train.py for standardized training.

model_train.py initializes the model and then iterates through each validation fold. It trains the model and times how long the training process takes, then evaluates the model with metrics provided in the training parameters using mutils.py. It then uses logger.py to serialize and save the model, along with diagnostic graphs and a report listing the model performance and the hyperparameters and training parameters used. 

#### Logging and Utilities
logger.py is provided information computed in model_train and makes a human-readable report outlining model performance and parameters used.

mutils.py provides tools to compute performance metrics (MSE, MAE, and more), do certain commonly used math operations, and make diagnostic graphs for logger.py.

#### Models
each model file contains a class that wraps around a scikit-learn or keras model. Included are CNN, lasso regression, kernel ridge regression, PCA + regression, Autoencoding + regression, random forest regression, and support vector regression. These wrappers are easy to write for a new model, and help de-clutter other parts of the training process.
Each model includes the following functions:

\_\_init\_\_ () which initializes the model

train() which trains the model on provided train X and y

predict() which makes predictions on provided data

diagnostics() which returns relevant diagnostic information about the model (such as weights, performance by epoch, etc.) to be recorded.
