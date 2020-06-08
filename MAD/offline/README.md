Offline Analysis
============================================================
This folder contains analysis on the following:
1. Metadata preparation - metadata includes artifacts needed for model training and online recommendation such as number of unique users, number of unique products, products that a user has interacted with etc.
2. Baseline Features creation - baseline features are features that will be used in model training and examples include number of user interactions, product interactions, days since earliest user interaction etc.
3. Data preparation - the prepared data would include steps such as mapping values of discrete variables to ids (example: mapping a uuid to a unique id) and adding the created baseline features to the individual part files which will be used as input in model training
4. Model utilities - includes utility class for data loading, network creation and training


Folder Structure
============================================================
1. nbs - contains the code for modelling experiments and actual model training
2. src - contains the code for metadata preparation, baseline feature creation, data preparation for model training and model utilities


To replicate the offline analysis
============================================================
0. Git clone the repo
1. Create a virtual environment with Python 3.6.5
2. Install the dependencies in pip-requirements.txt with "pip install -r pip-requirements.txt"
3. Follow the steps in the makefile to prepare data
4. Step into the nbs folder and run the binning_users_and_items.ipynb to bin users into segments
5. Finally run the notebooks viz. model_segGE20.ipynb and model_segLT20.ipynb for training the models for the segments
