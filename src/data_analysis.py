import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import scale, StandardScaler

from algorithms import Algorithms


class DataAnalysis:
    def __init__(self, verbose=0):
        self.predictions = None
        self.verbose = verbose

    def print(self, data):
        print('desc', data.describe())
        print('cols', data.columns)

    def splitXY(self, data):
        # separate X and y
        X = data.drop('normality', axis=1)
        y = data['normality']

        # scale X as part of preprocessing
        xScaled = StandardScaler().fit_transform(X)
        X = pd.DataFrame(xScaled, columns=X.columns)

        return X, y

    def splitTrainTest(self, X, y, trainSize=0.8, randomSeed=1):
        # split data into training and testing set
        xTrain, xTest, yTrain, yTest = train_test_split(
            X, y, test_size=1-trainSize, random_state=randomSeed)

        print('### Number of train samples:', xTrain.shape[0])

        return xTrain, xTest, yTrain, yTest

    def getDataCharacteristics(self, yTrain, yTest):
        # get all unique values (classes) and their count
        trainClassesCount = yTrain.value_counts(sort=False)
        testClassesCount = yTest.value_counts(sort=False)

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

    def predict(self, xTrain, xTest, yTrain, yTest, model):
        # k=5 cross validation on training set
        trainScores = cross_validate(
            model, xTrain, yTrain, scoring=('accuracy', 'balanced_accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted'), return_train_score=True)

        trainScores = [trainScores['test_accuracy'].mean(), trainScores['test_accuracy'].std(), trainScores['test_balanced_accuracy'].mean(),
                       trainScores['test_f1_weighted'].mean(), trainScores['test_precision_weighted'].mean(), trainScores['test_recall_weighted'].mean()]

        print('Training set:', trainScores)

        # fit the model
        model.fit(xTrain, yTrain)

        # predict
        yPredicted = model.predict(xTest)

        # get testing set scores
        acc = accuracy_score(yTest, yPredicted)
        balancedAcc = balanced_accuracy_score(yTest, yPredicted)
        f1 = f1_score(yTest, yPredicted, average='weighted')
        precision = precision_score(yTest, yPredicted, average='weighted')
        recall = recall_score(yTest, yPredicted, average='weighted')

        testScores = [acc, balancedAcc, f1, precision, recall]

        print('Testing set:', testScores)
        print('Unique predicted values:', np.unique(yPredicted))

        return trainScores, testScores

    def getScores(self, xTrain, xTest, yTrain, yTest, trainSize=1, randomSeeds=[1, 2, 3, 4, 5], selectedAlgorithms=['logReg', 'svm', 'dt', 'rf', 'ann']):
        algs = Algorithms()
        logReg = algs.logisticRegression()
        svm = algs.SVM()
        dt = algs.DecisionTree()
        rf = algs.RandomForest()
        ann = algs.ANN(epochs=5)

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
            noOfSamples, trainScores, testScores = self.calculateAverageScores(
                xTrain, xTest, yTrain, yTest, alg, algName, trainSize, randomSeeds)
            predictions[algName] = (trainScores, testScores)

            print('\nAverage train/test accuracy:',
                  trainScores[0], testScores[0], '\n')

        return noOfSamples, predictions

    def calculateAverageScores(self, xTrain, xTest, yTrain, yTest, model, algName, trainSize, randomSeeds):
        trainScoresAll = []
        testScoresAll = []
        noOfSamples = 0

        print('\n -->', algName, ':')

        for (idx, seed) in enumerate(randomSeeds):
            print('\nRun #' + str(idx + 1))

            # get data subset (if requested train dataset size is smaller than 1, split it again)
            xTrainSmall = xTrain
            yTrainSmall = yTrain

            if trainSize != 1:
                xTrainSmall, _, yTrainSmall, _ = self.splitTrainTest(
                    xTrain, yTrain, trainSize=trainSize, randomSeed=seed)

            # get prediction scores
            trainScores, testScores = self.predict(
                xTrainSmall, xTest, yTrainSmall, yTest, model)

            # save size of training data
            noOfSamples = xTrainSmall.shape[0]

            # append scores of current run
            trainScoresAll.append(trainScores)
            testScoresAll.append(testScores)

        # return number of samples in training set, average scores of all runs (train / test)
        return noOfSamples, np.average(np.array(trainScoresAll), axis=0).tolist(), np.average(np.array(testScoresAll), axis=0).tolist()
