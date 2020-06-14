Online App
============================================================
This folder contains code for creating and starting the app written in tornado


Folder Structure
============================================================
1. nbs - exploratory analysis which is not needed for running the app
2. src - contains the code for app creation


How to Start
============================================================
0. Git clone the repo
1. Create a virtual environment with Python 3.8.3
2. Install the dependencies in pip-requirements.txt with "pip install -r pip-requirements.txt"
3. Run the artifacts_for_recommendation.py file as "python artifacts_for_recommendation.py all"
4. Run the moving_data_to_db.py file as "python moving_data_to_db.py all" to move part of the data to redis 
5. Start the app with "python app.py"
