import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale, StandardScaler

from keras.wrappers.scikit_learn import KerasClassifier
from ann import ANN


class DataAnalysis:
    def __init__(self, data, testSize=0.2, verbose=0):
        self.data = data
        self.predictions = None
        self.verbose = verbose

        # separate X and y
        X = data.drop('normality', axis=1)
        y = data['normality']

        # scale the X as part of preprocessing
        xScaled = StandardScaler().fit_transform(X)
        X = pd.DataFrame(xScaled, columns=X.columns)

        # split data into training and testing set
        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(
            X, y, test_size=testSize, random_state=1)

        print('\n ### Number of samples:', self.xTrain.shape[0])

    def print(self, data):
        print('desc', data.describe())
        print('cols', data.columns)

    def getDataCharacteristics(self):
        # get all unique values (classes) and their count
        trainClassesCount = self.yTrain.value_counts(sort=False)
        testClassesCount = self.yTest.value_counts(sort=False)

        # save count for every class
        allClasses = [0, 1, 2, 3, 4, 5, 6, 7]
        classesCount = []
        countSum = 0
        for c in allClasses:
            # train classes
            trainCount = 0
            if c in trainClassesCount:
                trainCount = trainClassesCount[c]

            # test classes
            testCount = 0
            if c in testClassesCount:
                testCount = testClassesCount[c]

            # add both counts to overall sum
            countSum += (trainCount + testCount)

            # append tuple (train count, test count) of a class to array of results
            classesCount.append((trainCount, testCount))

        # append overall sum
        classesCount.append(countSum)

        return classesCount

    def logisticRegression(self):
        return LogisticRegression(
            verbose=self.verbose, max_iter=1000, n_jobs=-1)

    def SVM(self):
        return SVC(verbose=self.verbose)

    def DecisionTree(self):
        return DecisionTreeClassifier()

    def RandomForest(self, nEstimators=100):
        return RandomForestClassifier(n_estimators=nEstimators, n_jobs=-1)

    def ANN(self, epochs=10):
        ann = ANN()
        estimator = KerasClassifier(
            build_fn=ann.getModel, epochs=epochs, verbose=1)
        return estimator

    def predict(self, model, algName):
        print('\n -->', algName, ':')

        # k=5 cross validation on training set
        trainScores = cross_validate(
            model, self.xTrain, self.yTrain, scoring=('accuracy', 'balanced_accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted'), return_train_score=True)

        trainScores = [trainScores['test_accuracy'].mean(), trainScores['test_accuracy'].std(), trainScores['test_balanced_accuracy'].mean(),
                       trainScores['test_f1_weighted'].mean(), trainScores['test_precision_weighted'].mean(), trainScores['test_recall_weighted'].mean()]

        print('Training set:', trainScores)

        # fit the model
        model.fit(self.xTrain, self.yTrain)

        # predict
        yPredicted = model.predict(self.xTest)

        # get testing set scores
        acc = accuracy_score(self.yTest, yPredicted)
        balancedAcc = balanced_accuracy_score(self.yTest, yPredicted)
        f1 = f1_score(self.yTest, yPredicted, average='weighted')
        precision = precision_score(self.yTest, yPredicted, average='weighted')
        recall = recall_score(self.yTest, yPredicted, average='weighted')

        testScores = [acc, balancedAcc, f1, precision, recall]

        print('Testing set:', testScores)
        print('Unique predicted values:', np.unique(yPredicted))

        return trainScores, testScores

    def getScores(self, selectedAlgorithms=['logReg', 'svm', 'dt', 'rf', 'ann']):
        logReg = self.logisticRegression()
        svm = self.SVM()
        dt = self.DecisionTree()
        rf = self.RandomForest()
        ann = self.ANN(epochs=5)

        # available algorithms ([name, implementation])
        algorithms = {
            'logReg': ['Logistic Regression', logReg],
            'svm': ['SVM', svm],
            'dt': ['Decision Tree', dt],
            'rf': ['Random Forest', rf],
            'ann': ['ANN', ann],
        }

        predictions = {}

        for selected in selectedAlgorithms:
            [algName, alg] = algorithms[selected]
            trainScores, testScores = self.predict(alg, algName)
            predictions[algName] = (trainScores, testScores)
            print(algName, 'train/test accuracy:',
                  trainScores[0], testScores[0])

        return self.xTrain.shape[0], predictions
