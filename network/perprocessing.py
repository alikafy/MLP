from abc import ABC

import pandas as pd
from sklearn import preprocessing


class InputNetwork(ABC):
    address_file: str

    def per_process(self):
        pass


class PerProcessingBankData(InputNetwork):
    def __init__(self, address_file):
        self.address_file = address_file
        self.data = pd.read_csv(address_file)
        self.data.columns = ['job', 'education', 'gender', 'minority']

    def per_process(self):
        inputs, targets = self.data.iloc[:, 0:3], self.data.iloc[:, 3]
        inputs = pd.get_dummies(inputs)

        scaled_data = preprocessing.StandardScaler().fit(inputs)
        scaled_data = scaled_data.transform(inputs)
        scaled_data = scaled_data.reshape(473, 1, 6)
        targets = targets.replace("yes", 1)
        targets = targets.replace("no", 0)
        return scaled_data, targets
