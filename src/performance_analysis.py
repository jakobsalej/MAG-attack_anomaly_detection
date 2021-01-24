import time
from datetime import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV

from algorithms import Algorithms


class PerformanceAnalysis:
    def __init__(self, resultsDir=datetime.now().strftime("%d-%m-%Y(%H-%M-%S)"), verbose=0):
        self.predictions = None
        self.verbose = verbose
        self.dirName = 'results/performance_analysis'
        self.resultsDirName = f'{self.dirName}/{resultsDir}'
        self.model = {}

        if not os.path.exists(self.resultsDirName):
            os.mkdir(self.resultsDirName)

        # available algorithms ([name, implementation])
        algs = Algorithms()
        self.algorithms = {
            'logReg': ['LR', algs.logisticRegression()],
            'svm': ['SVM', algs.SVM()],
            'dt': ['DT', algs.DecisionTree()],
            'rf': ['RF', algs.RandomForest()],
            'ann': ['ANN', algs.ANN()],
        }

    def measureFitTime(self, alg, X, y, repeats=10):
        times = []
        calibratedModel = None

        for i in range(repeats):
            calibratedModel = None
            # Calibrated Classifier uses 5-fold CV by default
            calibratedModel = CalibratedClassifierCV(
                base_estimator=self.algorithms[alg][1])

            # fit the model
            startTime = time.time()
            calibratedModel.fit(X, y)
            duration = time.time() - startTime
            print(f'{alg} train, run {i+1}: {duration}s')
            times.append(duration)

        # Keep last trained model for predictions
        self.model[alg] = calibratedModel

        return times

    def measurePredictTime(self, alg, trainX, trainY, X, y, repeats=10):
        times = []

        # Check if train model for selected alg exists
        if not self.model[alg]:
            calibratedModel = CalibratedClassifierCV(
                base_estimator=self.algorithms[alg][1])
            calibratedModel.fit(X, y)
            self.model[alg] = calibratedModel

        for i in range(repeats):
            startTime = time.time()
            _ = self.model[alg].predict(X)
            duration = time.time() - startTime
            print(f'{alg} predict, run {i+1}: {duration}s')
            times.append(duration)

        return times

    def readFile(self, path):
        data = pd.read_csv(path, index_col=0)
        X = data.drop('normality', axis=1)
        y = data['normality']
        return X, y

    def savePlot(self, data, fileName):
        sns.set()
        plt.figure()
        plot = sns.barplot(data=data)
        plot.set(xlabel='Algorithms', ylabel='Time [s]')
        plot.get_figure().savefig(f'{self.resultsDirName}/{fileName}.png')

    def saveDataToCSV(self, data, fileName):
        data.to_csv(f'{self.resultsDirName}/{fileName}.csv')


if __name__ == '__main__':
    pa = PerformanceAnalysis()

    # load testing set
    dataFolder = 'results/mode_1/08-12-2020(00-58-51)/datasets'
    testX, testY = pa.readFile(f'{dataFolder}/AD_set_test.csv')

    # selected algs & sizes
    algs = ['logReg', 'svm', 'dt', 'rf', 'ann']
    # algs = ['logReg', 'dt']
    datasetSizes = [20, 40, 60, 80, 100]
    # datasetSizes = [20, 40, 60]

    # For charts
    averageFitTimes = {}
    averagePredictTimes = {}

    for alg in algs:
        algTag = pa.algorithms[alg][0]

        averageFitTimes[alg] = {}
        averagePredictTimes[alg] = {}
        fitTimes = {}
        predictTimes = {}

        for size in datasetSizes:
            # load training set (testing set is always the same)
            trainDatasetFile = f'{dataFolder}/AD_set_train{size}_seed42.csv'
            if size == 100:
                trainDatasetFile = f'{dataFolder}/AD_set_train.csv'

            trainX, trainY = pa.readFile(trainDatasetFile)

            # measure train time
            measuredFitTimes = pa.measureFitTime(alg, trainX, trainY)
            fitTimes[size] = measuredFitTimes
            averageFitTimes[alg][size] = np.average(measuredFitTimes)

            # measure predict time
            measuredPredictTimes = pa.measurePredictTime(
                alg, trainX, trainY, testX, testY)
            predictTimes[size] = measuredPredictTimes
            averagePredictTimes[alg][size] = np.average(measuredPredictTimes)

        # Save alg results to a file
        pa.saveDataToCSV(pd.DataFrame(fitTimes), f'fit_times_{algTag}')
        pa.saveDataToCSV(pd.DataFrame(predictTimes), f'predict_times_{algTag}')

    # save times to .csv
    averageFitTimesDF = pd.DataFrame(averageFitTimes)
    averagePredictTimesDF = pd.DataFrame(averagePredictTimes)
    pa.saveDataToCSV(averageFitTimesDF, 'avg_fit_times')
    pa.saveDataToCSV(averagePredictTimesDF, 'avg_predict_times')
