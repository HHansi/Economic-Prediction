# Created by Hansi at 11/22/2020

fall_label = -1
static_label = 0
growth_label = 1

random_seed = 157

f1_label = "f1"
recall_label = "recall"
precision_label = "precision"

# future n prediction = train_test_series_length - train_series_length

data_processing_args = {
    'column_names': ['Data'],  # ['GDP_CVM', 'Service_T', 'Production_T', 'Construction_SA_CVM']
    'test_size': 0.3,
    'normalise': True,
    'train_test_series_length': 5,
    'train_series_length': 4
}

saxvsm_args = {
    'n_bins': 3,
    'word_size': 3,
    'window_size': 3
}
