## How to run the script

1. Create virtual environment with Python 3.10.16
2. Install the requirements in the virtual environment with `pip install -r requirements.txt`
3. Update the environment variable `MODEL_NAME` in `run.sh` with the model that needs to be trained / tested.
4. Then, run:
    - `make train` for training
    - `make test` for testing
    - `make validate` for validation