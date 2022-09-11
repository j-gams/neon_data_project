# MODEL TRAINING AND EVALUATION PIPELINE
## Overview
The general structure of the training/evaluation pipeline is as follows:
#### Data Management
Two files in the create_data directory are used to manage data for the training and evaluation process.
These interact with the directory and file structure of the dataset to greatly streamline loading and using the dataset.
dat_obj.py is a dataset object. Initializing this with a dataset will load all the splits and folds and will store both the raw metadata csv and a data fold object for each.
