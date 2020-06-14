help:
	@echo "    artifacts-for-recommendation"
	@echo "        Creates metadata needed for recommendation"
	@echo "    moving-data-to-db"
	@echo "        Moves data in json files to redis"
	@echo "    run-app-locally"
	@echo "        Runs the app locally"
	@echo "    run-app-docker"
	@echo "        Runs the app in docker"

artifacts-for-recommendation:
	python src/artifacts_for_recommendation.py all

moving-data-to-db:
	python src/moving_data_to_db.py all

run-app-locally:
	python src/app.py

run-app-docker:
	docker build -f Dockerfile -t recommendation:latest .&
	docker run -p 8881:8880 recommendation
