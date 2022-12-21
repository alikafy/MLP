import pandas as pd
import numpy as np
from sklearn import preprocessing


def preprocessing_raw_data(data_path: str = 'BankWages.csv'):
    my_data = pd.read_csv(data_path, delimiter=',')
    my_data.columns = ['job', 'education', 'gender', 'minority']
    data_without_labels = my_data.filter(items=['job', 'education', 'gender'])
    scaled_data = preprocessing.StandardScaler().fit(data_without_labels)
    scaled_data = scaled_data.transform(data_without_labels)
    return np.hstack((scaled_data, my_data.filter(items=['minority'])))  # add column label
