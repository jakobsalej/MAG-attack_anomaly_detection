import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale, StandardScaler


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

        print('Number of samples:', self.xTrain.shape[0])

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

        algs = [logReg, svm, dt, rf]
        algNames = ['Logistic Regression', 'SVM',
                    'Decision Tree', 'Random Forest']

        predictions = {}

        for alg, algName in zip(algs, algNames):
            trainScore, testScore = self.predict(alg)
            predictions[algName] = [trainScore, testScore]

            print(algName, 'train/test score:', trainScore, testScore)

        return self.xTrain.shape[0], predictions
