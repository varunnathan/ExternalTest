## Overall design
1. `/src_may3/main` contains the codebase for the entire project
2. `/src_may3/config` contains training-related parameters for e2e
3. `/src_may3/requirements.txt` contains the python requirements for the project
---
## How to run?
1. Make sure you are on python 3.12 (My python version is python 3.12.8)
2. Create a virtual environment => `python -m venv /py3128_external_test`
3. Install the requirements in the virtual environment => `pip install -r requirements.txt`
2. From the root folder of the project run the following commands
    > `PYTHONPATH=./ python main/startup.py --module cross_domain_slot_prediction`
