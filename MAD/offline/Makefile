help:
	@echo "    metadata"
	@echo "        Creates meta files needed for model training and online inference"
	@echo "    baseline-features"
	@echo "        Creates the baseline features needed for training the model"
	@echo "    data-preparation"
	@echo "        Prepares the data for model training"

metadata:
	python src/metadata_utils all

baseline-features:
	python src/baseline_feats_utils

data-preparation:
	python src/data_preparation
