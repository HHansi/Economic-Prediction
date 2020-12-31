# Created by Hansi at 11/22/2020
fall_label = 0
static_label = 1
growth_label = 2

random_seed = 157

f1_label = "f1"
recall_label = "recall"
precision_label = "precision"

data_file_path = "../../data/Data.csv"
result_file_path = "../../results/saxsvm_8.tsv"

univariate_data_processing_args = {
    'column_name': 'GDP_CVM',
    'test_size': 0.2,
    'normalise': True,
    'train_test_series_length': 6, #5
    'train_series_length': 4 #4
}

multivariate_data_processing_args = {
    'column_names': ['GDP_CVM'], # ['GDP_CVM', 'Service_T', 'Production_T', 'Construction_SA_CVM']
    'test_size': 0.2,
    'normalise': True,
    'train_test_series_length': 13,  # 5
    'train_series_length': 8 # 4
}

saxvsm_args = {
    'n_bins': 2,
    'word_size': 5,
    'window_size': 8
}
