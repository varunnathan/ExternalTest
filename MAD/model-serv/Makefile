help:
	@echo "    begin"
	@echo "        Creates the model archive and starts the torchserve service"
	@echo "    create-model-archive"
	@echo "        Creates a .mar file for the ckpt or state_dict model object"
	@echo "    start-service"
	@echo "        Starts the torchserve service"
	@echo "    stop-service"
	@echo "        Stops the torchserve service"

begin:
	make create-model-archive&
	make start-service

create-model-archive:
	cd /Users/varunn/Documents/ExternalTest_Data/MAD/ && echo "root folder of MAD" && \
	torch-model-archiver --model-name Class_model_SegGE20 --version 1.0 \
	--serialized-file model/Class_model_SegGE20_E1_ckpt.pt \
	--handler userevent_classifier_torchserve_handler.py --export-path model-store \
	--extra-files /Users/varunn/Documents/ExternalTest/MAD/offline/src/network.py,/Users/varunn/Documents/ExternalTest/MAD/offline/src/constants.py,/Users/varunn/Documents/ExternalTest/MAD/offline/src/model_utils.py,/Users/varunn/Documents/ExternalTest/MAD/offline/src/model_params.py

start-service:
	torchserve --start --model-store model-store --models Class_model_SegGE20=Class_model_SegGE20.mar --ts-config config.properties

stop-service:
	torchserve --stop
