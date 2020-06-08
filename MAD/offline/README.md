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


Note: Makefile contains the steps to recreate all the files generated in the offline analysis
