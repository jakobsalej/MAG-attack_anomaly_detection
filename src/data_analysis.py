from algorithms import Algorithms
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.combine import SMOTEENN
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import scale, StandardScaler, label_binarize
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
import matplotlib.pyplot as plt
import os

import numpy as np
import pandas as pd
from collections import Counter

import matplotlib
matplotlib.use('Agg')


np.set_printoptions(precision=4)


class DataAnalysis:
    def __init__(self, dirName, mode, pi=False, verbose=0):
        self.predictions = None
        self.pi = pi
        self.verbose = verbose
        self.dirName = dirName
        self.mode = mode
        self.targetClasses = [0, 1, 2, 3, 4, 5, 6, 7]
        self.classDistribution = {}

        # available algorithms ([name, implementation])
        algs = Algorithms(pi=self.pi)
        self.algorithms = {
            'logReg': ['LR', algs.logisticRegression()],
            'svm': ['SVM', algs.SVM()],
            'dt': ['DT', algs.DecisionTree()],
            'rf': ['RF', algs.RandomForest()],
            'ann': ['ANN', algs.ANN()],
        }

    def saveData(self, data, fileName, folder='datasets'):
        try:
            if not os.path.exists(f'{self.dirName}/{folder}'):
                os.mkdir(f'{self.dirName}/{folder}')
            data.to_csv(f'{self.dirName}/{folder}/{fileName}.csv', index=False)
        except Exception as exception:
            print(exception)

    def print(self, data):
        print('desc', data.describe())
        print('cols', data.columns)

    def randomResample(self, X, y, dropPercentage=0.90, randomSeed=42):
        data = X
        data['normality'] = y.values

        print('Original training set shape %s' % Counter(y))

        # randomly sample examples of class 7 to drop
        samplesToDrop = data.query('normality == 7').sample(
            frac=dropPercentage, random_state=randomSeed)
        data.drop(samplesToDrop.index, inplace=True)

        resY = data['normality']
        resX = data.drop('normality', axis=1)

        print('Resampled training set shape %s' % Counter(resY))

        return resX, resY

    def resample(self, X, y):
        cnn = CondensedNearestNeighbour(
            sampling_strategy='majority', random_state=42, n_jobs=4)
        sme = SMOTEENN(sampling_strategy='not majority',
                       random_state=42, n_jobs=4)

        print('Original training set shape %s' % Counter(y))

        # first undersample majority class
        resX, resY = cnn.fit_resample(X, y)

        # then oversample all others
        resX, resY = sme.fit_resample(resX, resY)

        print('Resampled training set shape %s' % Counter(resY))

        return resX, resY

    def splitXY(self, data):
        # separate X and y
        X = data.drop('normality', axis=1)
        y = data['normality']

        X = pd.DataFrame(X, columns=X.columns)

        return X, y

    def splitTrainTest(self, X, y, trainSize, randomSeed, scale=False, resample=False):
        # split data into training and testing set
        if trainSize == 1:
            return X, None, y, None

        xTrain, xTest, yTrain, yTest = train_test_split(
            X, y, test_size=1-trainSize, random_state=randomSeed)

        # scale X as part of preprocessing
        if scale:
            # fit Standard Scaler on train data
            standardScaler = StandardScaler().fit(xTrain)

            # transform xTrain
            xTrainColumns = xTrain.columns
            xTrain = standardScaler.transform(xTrain)
            xTrain = pd.DataFrame(xTrain, columns=xTrainColumns)

            # transform xTest
            xTestColumns = xTest.columns
            xTest = standardScaler.transform(xTest)
            xTest = pd.DataFrame(xTest, columns=xTestColumns)

        if resample:
            xTrain, yTrain = self.randomResample(xTrain, yTrain)

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

    def saveClassDistribution(self, fileName, data):
        # save class distribution for each file
        c = Counter(data)
        allExamples = data.shape[0]
        self.classDistribution[fileName] = [
            f'{c[targetClass]} ({(c[targetClass] / allExamples) * 100:0.2f}%)' for targetClass in self.targetClasses]

    def getClassDistribution(self):
        return pd.DataFrame.from_dict(self.classDistribution, orient='index')

    def saveConfusionMatrix(self, yTrue, yPredicted, fileName):
        # save to .csv
        pd.DataFrame(confusion_matrix(yTrue, yPredicted)).to_csv(
            f'{self.dirName}/results/CM_{fileName}.csv', index=False)

        # save graph
        plt.close('all')
        plt.figure()
        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix(yTrue, yPredicted, normalize=None), display_labels=self.targetClasses)
        disp = disp.plot(include_values=True, cmap=plt.cm.Blues,
                         ax=None, xticks_rotation='horizontal')
        disp.ax_.set_title(f'CM_{fileName}_M{self.mode}')
        disp.figure_.savefig(f'{self.dirName}/graphs/CM_{fileName}.png')

    def saveROC(self, model, xTrain, yTrain, xTest, yTest, fileName):
        # Binarize the y
        yTrain = label_binarize(yTrain, classes=self.targetClasses)
        yTest = label_binarize(yTest, classes=self.targetClasses)
        nClasses = yTest.shape[1]

        # Learn to predict each class against the other
        classifier = OneVsRestClassifier(model)
        yPredicted = classifier.fit(xTrain, yTrain).predict(xTest)

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
        plt.title(f'ROC_{fileName}_M{self.mode}')
        plt.legend(loc="lower right")
        plt.savefig(f'{self.dirName}/graphs/ROC_{fileName}.png')

    def predict(self, xTrain, xTest, yTrain, yTest, model, fileName, testFileName):
        # Calibrated Classifier uses 5-fold CV by default
        calibratedModel = CalibratedClassifierCV(base_estimator=model)

        # fit the model
        calibratedModel.fit(xTrain, yTrain)

        # predict on train data
        yTrainPredicted = calibratedModel.predict(xTrain)
        acc = accuracy_score(yTrain, yTrainPredicted)
        balancedAcc = balanced_accuracy_score(yTrain, yTrainPredicted)
        f1 = f1_score(yTrain, yTrainPredicted,
                      average='weighted', zero_division=0)
        precision = precision_score(
            yTrain, yTrainPredicted, average='weighted', zero_division=0)
        recall = recall_score(
            yTrain, yTrainPredicted, average='weighted', zero_division=0)

        trainScores = [acc, 0, balancedAcc, f1, precision, recall]
        print('Training set:', trainScores)

        # predict on test data
        yPredicted = calibratedModel.predict(xTest)

        # save roc graphs
        self.saveROC(model, xTrain, yTrain,
                     xTest, yTest, testFileName)

        # save confusion matrix to .csv
        self.saveConfusionMatrix(yTest, yPredicted, testFileName)

        # save testing set (with target and predicted values) to .csv
        self.saveData(xTest.assign(normality=yTest.values, predicted=yPredicted),
                      f'AD_set_test{testFileName}')

        # get testing set scores
        acc = accuracy_score(yTest, yPredicted)
        balancedAcc = balanced_accuracy_score(yTest, yPredicted)
        f1 = f1_score(yTest, yPredicted, average='weighted', zero_division=0)
        precision = precision_score(
            yTest, yPredicted, average='weighted', zero_division=0)
        recall = recall_score(
            yTest, yPredicted, average='weighted', zero_division=0)

        testScores = [acc, balancedAcc, f1, precision, recall]

        print('### Number of test samples:', xTest.shape[0])
        print('Testing set:', testScores)

        return trainScores, testScores

    def getScores(self, xTrain, xTest, yTrain, yTest, trainSize=1, randomSeeds=[1, 2, 3, 4, 5], selectedAlgorithms=None, mode=1):
        # by default, select all algorithms
        if selectedAlgorithms is None:
            selectedAlgorithms = self.algorithms.keys()

        predictions = {}

        # get scores for each algorithm
        for selected in selectedAlgorithms:
            algorithm = self.algorithms[selected]
            noOfSamples, trainScores, testScores = self.calculateAverageScores(
                xTrain, xTest, yTrain, yTest, algorithm, trainSize, randomSeeds, mode)
            predictions[algorithm[0]] = (trainScores, testScores)

            print('\nAverage train/test accuracy:',
                  trainScores[0], testScores[0])
            print('Average train/test balanced accuracy:',
                  trainScores[2], testScores[1], '\n')

        return noOfSamples, predictions

    def calculateAverageScores(self, xTrain, xTest, yTrain, yTest, algorithm, trainSize, randomSeeds, mode):
        trainScoresAll = []
        testScoresAll = []
        noOfSamples = 0

        [algName, alg] = algorithm
        print('\n -->', algName, ':')

        # repeat multiple times if different seeds specified
        for (idx, seed) in enumerate(randomSeeds):
            print('\nRun #' + str(idx + 1))

            # if training set for a given size and random seed is not saved yet, generate it and save it
            fileName = f'AD_set_train{trainSize * 100:.0f}_seed{seed}'

            if mode == 1:
                # generate new subsets
                if trainSize != 1 and not os.path.exists(f'{self.dirName}/datasets/{fileName}.csv'):
                    # generate new training set
                    xTrain, _, yTrain, _ = self.splitTrainTest(
                        xTrain, yTrain, trainSize=trainSize, randomSeed=seed)

                    # save new training set
                    self.saveData(xTrain.assign(
                        normality=yTrain.values), fileName)

                    # save class distribution
                    self.saveClassDistribution(fileName, yTrain)

                else:
                    # TODO: read from existing file instead of generating set again
                    # use existing training set
                    xTrain, _, yTrain, _ = self.splitTrainTest(
                        xTrain, yTrain, trainSize=trainSize, randomSeed=seed)

            elif not os.path.exists(f'{self.dirName}/datasets/{fileName}.csv'):
                # mode 0, keep sets the same and save them
                self.saveData(xTrain.assign(
                    normality=yTrain.values), fileName)

            # get prediction scores
            trainScores, testScores = self.predict(
                xTrain, xTest, yTrain, yTest, alg, fileName, f'{trainSize * 100:.0f}_{algName}')

            # save size of training data
            noOfSamples = xTrain.shape[0]

            # append scores of current run
            trainScoresAll.append(trainScores)
            testScoresAll.append(testScores)

            # if train size is 1, there is no point in repeating and calculating averages, as data is always the same
            if trainSize == 1:
                break

        # return number of samples in training set, average scores of all runs (train / test)
        return noOfSamples, np.average(np.array(trainScoresAll), axis=0).tolist(), np.average(np.array(testScoresAll), axis=0).tolist()
