## Data Preparation Steps
1. Download Amazon review datasets from http://jmcauley.ucsd.edu/data/amazon/
2. Index datasets: python ./index_and_filter_review_data.py /Users/varunnathan/Documents/Personal/interview_external/whatfix/training/data/raw/Cell_Phones_and_Accessories_5.json.gz /Users/varunnathan/Documents/Personal/interview_external/whatfix/training/data/preprocessed/ 5
3. Extract queries and Split train/test
    1. Download the meta data from http://jmcauley.ucsd.edu/data/amazon/ 
    2. Match the meta data with the indexed data: python ./meta_data_matching.py false /Users/varunnathan/Documents/Personal/interview_external/whatfix/training/data/raw/meta_Cell_Phones_and_Accessories_v4.json.gz /Users/varunnathan/Documents/Personal/interview_external/whatfix/training/data/preprocessed/min_count5
    3. Split datasets for training and test: python ./split_train_and_test_data.py /Users/varunnathan/Documents/Personal/interview_external/whatfix/training/data/preprocessed/min_count5/ 0.2 0.3