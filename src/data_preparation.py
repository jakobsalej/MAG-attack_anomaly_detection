import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, scale, StandardScaler


class DataPreparation:
    def __init__(self, path):
        self.data = pd.read_csv(path)
        self.normalityMapping = None

    def print(self):
        print(self.data.describe())
        print(self.data.columns)
        print(self.data.shape)

        # output all columns and their unique values
        for column in self.data.columns:
            print(column, self.data[column].unique())

    def prepareData(self):
        # replace 'nan' values in column 'accessedNodeType' with ‘Malicious’
        self.data['accessedNodeType'] = self.data['accessedNodeType'].fillna(
            'Malicious')

        # replace all non numeric values in column 'value'
        self.data = self.data.replace(
            to_replace={'value': {'False': '0.0', 'false': '0.0', 'True': '1.0', 'true': '1.0', 'Twenty': '20.0', 'twenty': '20.0', 'none': '0.0', 'None': '0.0'}})
        self.data['value'] = pd.to_numeric(self.data['value'], errors='coerce')

        # set missing values to 0
        self.data['value'].fillna(0, inplace=True)

        # remove 'timestamp' column
        self.data = self.data.drop(columns=['timestamp'])

        # apply label encoding
        # use label enconding on all columns except column 'value' and 'normality'
        cols = [col for col in self.data.columns if col not in [
            'value', 'normality']]
        self.data[cols] = self.data[cols].apply(LabelEncoder().fit_transform)

        # seperatly use label encoding on 'normality', so we can get mappings
        le = LabelEncoder()
        le.fit(self.data['normality'])
        self.data['normality'] = le.transform(self.data['normality'])
        self.normalityMapping = dict(
            zip(le.transform(le.classes_), le.classes_))

    def getNormalityMapping(self):
        return self.normalityMapping

    def returnData(self, fraction=1, randomSeed=1):
        # return fraction of all data (or all, if fraction is 1) randomly
        # use random_state to ensure the reproducibility of the examples
        return self.data.sample(frac=fraction, random_state=randomSeed)
