import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import scale, StandardScaler, label_binarize
from sklearn.multiclass import OneVsRestClassifier

from algorithms import Algorithms


np.set_printoptions(precision=4)


class DataAnalysis:
    def __init__(self, dirName, verbose=0):
        self.predictions = None
        self.verbose = verbose
        self.dirName = dirName
        self.outputClasses = [0, 1, 2, 3, 4, 5, 6, 7]

        # available algorithms ([name, implementation])
        algs = Algorithms()
        self.algorithms = {
            'logReg': ['Logistic Regression', algs.logisticRegression()],
            'svm': ['SVM', algs.SVM()],
            'dt': ['Decision Tree', algs.DecisionTree()],
            'rf': ['Random Forest', algs.RandomForest()],
            'ann': ['ANN', algs.ANN(epochs=5)],
        }

    def saveData(self, data, fileName, folder='datasets'):
        try:
            if not os.path.exists(f'{self.dirName}/{folder}'):
                os.mkdir(f'{self.dirName}/{folder}')
            data.to_csv(f'{self.dirName}/{folder}/{fileName}.csv')
        except Exception as exception:
            print(exception)

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

    def saveConfusionMatrix(self, yTrue, yPredicted, fileName):
        # save to .csv
        pd.DataFrame(confusion_matrix(yTrue, yPredicted)).to_csv(
            f'{self.dirName}/results/CM_{fileName}.csv')

        # save graph
        plt.close('all')
        plt.figure()
        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix(yTrue, yPredicted, normalize='true'), display_labels=self.outputClasses)
        disp = disp.plot(include_values=True, cmap=plt.cm.Blues,
                         ax=None, xticks_rotation='horizontal')
        disp.ax_.set_title(fileName)
        disp.figure_.savefig(f'{self.dirName}/graphs/CM_{fileName}.png')

    def saveROC(self, model, xTrain, yTrain, xTest, yTest, fileName):
        # Binarize the y
        yTrain = label_binarize(yTrain, classes=self.outputClasses)
        yTest = label_binarize(yTest, classes=self.outputClasses)
        nClasses = yTest.shape[1]

        # Learn to predict each class against the other
        classifier = OneVsRestClassifier(model)
        yPredicted = classifier.fit(xTrain, yTrain).predict_proba(xTest)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        rocAuc = dict()
        for i in range(nClasses):
            fpr[i], tpr[i], _ = roc_curve(yTest[:, i], yPredicted[:, i])
            rocAuc[i] = auc(fpr[i], tpr[i])

        # Plot all ROC curves
        plt.close('all')
        plt.figure()
        lw = 2
        for i in range(nClasses):
            plt.plot(fpr[i], tpr[i], lw=lw,
                     label='Class {0} (area = {1:0.2f})'
                     ''.format(i, rocAuc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(fileName)
        plt.legend(loc="lower right")
        plt.savefig(f'{self.dirName}/graphs/ROC_{fileName}.png')

    def predict(self, xTrain, xTest, yTrain, yTest, model, fileName, testFileName):
        # k=5 cross validation on training set
        trainScores = cross_validate(
            model, xTrain, yTrain, scoring=('accuracy', 'balanced_accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted'), return_train_score=True, return_estimator=True)

        estimators = trainScores['estimator']
        trainScores = [trainScores['test_accuracy'].mean(), trainScores['test_accuracy'].std(), trainScores['test_balanced_accuracy'].mean(),
                       trainScores['test_f1_weighted'].mean(), trainScores['test_precision_weighted'].mean(), trainScores['test_recall_weighted'].mean()]

        print('Training set:', trainScores)

        # fit the model
        model.fit(xTrain, yTrain)

        # predict
        yPredicted = model.predict(xTest)

        # save roc graphs
        self.saveROC(model, xTrain, yTrain, xTest, yTest, testFileName)

        # save confusion matrix to .csv
        self.saveConfusionMatrix(yTest, yPredicted, testFileName)

        # save testing set (with target and predicted values) to .csv
        self.saveData(xTest.assign(normality=yTest.values, predicted=yPredicted),
                      f'AD_set_test{testFileName}')

        # get testing set scores
        acc = accuracy_score(yTest, yPredicted)
        balancedAcc = balanced_accuracy_score(yTest, yPredicted)
        f1 = f1_score(yTest, yPredicted, average='weighted')
        precision = precision_score(yTest, yPredicted, average='weighted')
        recall = recall_score(yTest, yPredicted, average='weighted')

        testScores = [acc, balancedAcc, f1, precision, recall]

        print('### Number of test samples:', xTest.shape[0])
        print('Testing set:', testScores)
        # print('Unique predicted values:', np.unique(yPredicted))

        return trainScores, testScores

    def getScores(self, xTrain, xTest, yTrain, yTest, trainSize=1, randomSeeds=[1, 2, 3, 4, 5], selectedAlgorithms=None):
        # by default, select all algorithms
        if selectedAlgorithms is None:
            selectedAlgorithms = self.algorithms.keys()

        predictions = {}

        # get scores for each algorithm
        for selected in selectedAlgorithms:
            algorithm = self.algorithms[selected]
            noOfSamples, trainScores, testScores = self.calculateAverageScores(
                xTrain, xTest, yTrain, yTest, algorithm, trainSize, randomSeeds)
            predictions[algorithm[0]] = (trainScores, testScores)

            print('\nAverage train/test accuracy:',
                  trainScores[0], testScores[0], '\n')

        return noOfSamples, predictions

    def calculateAverageScores(self, xTrain, xTest, yTrain, yTest, algorithm, trainSize, randomSeeds):
        trainScoresAll = []
        testScoresAll = []
        noOfSamples = 0

        [algName, alg] = algorithm

        print('\n -->', algName, ':')

        for (idx, seed) in enumerate(randomSeeds):
            print('\nRun #' + str(idx + 1))

            # if training set for a given size and random seed is not saved yet, generate it and save it
            fileName = f'AD_set_train{trainSize * 100:.0f}_seed{seed}'

            if trainSize != 1 and not os.path.exists(f'{self.dirName}/datasets/{fileName}.csv'):
                # generate new training set
                xTrainSmall, _, yTrainSmall, _ = self.splitTrainTest(
                    xTrain, yTrain, trainSize=trainSize, randomSeed=seed)

                # save new training set
                self.saveData(xTrainSmall.assign(
                    normality=yTrainSmall.values), fileName)

            else:
                # TODO: read from existing file instead of generating set again
                # use existing training set
                xTrainSmall, _, yTrainSmall, _ = self.splitTrainTest(
                    xTrain, yTrain, trainSize=trainSize, randomSeed=seed)

            # get prediction scores
            trainScores, testScores = self.predict(
                xTrainSmall, xTest, yTrainSmall, yTest, alg, fileName, f'{trainSize * 100:.0f}_{algName}')

            # save size of training data
            noOfSamples = xTrainSmall.shape[0]

            # append scores of current run
            trainScoresAll.append(trainScores)
            testScoresAll.append(testScores)

            # if train size is 1, there is no point in repeating and calculating averages, as data is always the same
            if trainSize == 1:
                break

        # return number of samples in training set, average scores of all runs (train / test)
        return noOfSamples, np.average(np.array(trainScoresAll), axis=0).tolist(), np.average(np.array(testScoresAll), axis=0).tolist()
