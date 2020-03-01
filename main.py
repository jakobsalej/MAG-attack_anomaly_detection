import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, scale, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# np.random.seed(10)


class DataPrediction:
    def __init__(self, data, testSize=0.2, verbose=0):
        self.data = data
        self.predictions = None
        self.verbose = verbose

        X = data.drop('normality', axis=1)

        # scale the X as part of preprocessing
        xScaled = StandardScaler().fit_transform(X)
        X = pd.DataFrame(xScaled, columns=X.columns)

        y = data['normality']

        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(
            X, y, test_size=testSize, random_state=0)

    def print(self, data):
        print('desc', data.describe())
        print('cols', data.columns)

    def prepareData(self):
        pass

    def logisticRegression(self):
        return LogisticRegression(
            verbose=self.verbose, max_iter=1000, n_jobs=-1)

    def SVM(self):
        return SVC(verbose=self.verbose)

    def DecisionTree(self):
        return DecisionTreeClassifier()

    def RandomForest(self, nEstimators=100):
        return RandomForestClassifier(n_estimators=nEstimators, n_jobs=-1)

    def predict(self, model):
        # k=5 cross validation on training set
        trainScores = cross_val_score(
            model, self.xTrain, self.yTrain, n_jobs=-1)

        # fit the model
        model.fit(self.xTrain, self.yTrain)

        # prediction
        testScore = model.score(self.xTest, self.yTest)
        return trainScores.mean(), testScore

    def getScores(self):
        logReg = self.logisticRegression()
        svm = self.SVM()
        dt = self.DecisionTree()
        rf = self.RandomForest()

        trainScore, testScore = self.predict(rf)

        print('Train/test score:', trainScore, testScore)


class DataAnalysis:
    def __init__(self, path):
        self.data = pd.read_csv(path)

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
        # use label enconding on all columns except column 'value'
        cols = [col for col in self.data.columns if col not in ['value']]
        self.data[cols] = self.data[cols].apply(LabelEncoder().fit_transform)

    def returnData(self, percentage=1):
        sampleData = self.data.sample(frac=percentage, random_state=1)
        print('Number of samples:', sampleData.shape[0])

        return sampleData


if __name__ == '__main__':
    # prepare the data
    da = DataAnalysis('data/mainSimulationAccessTraces.csv')
    da.prepareData()
    # data.print()

    # Run predictions
    sampleData = da.returnData(0.15)
    dp = DataPrediction(sampleData)
    dp.getScores()
