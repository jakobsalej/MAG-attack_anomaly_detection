import time
from datetime import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from algorithms import Algorithms


class PerformanceAnalysis:
    def __init__(self, dirName, resultsDir=datetime.now().strftime("%d-%m-%Y(%H-%M-%S)"), verbose=0):
        self.predictions = None
        self.verbose = verbose
        self.dirName = dirName
        self.resultsDirName = f'{dirName}/{resultsDir}'

        if not os.path.exists(self.resultsDirName):
            os.mkdir(self.resultsDirName)

        # available algorithms ([name, implementation])
        algs = Algorithms()
        self.algorithms = {
            'logReg': ['Logistic Regression', algs.logisticRegression()],
            'svm': ['SVM', algs.SVM()],
            'dt': ['Decision Tree', algs.DecisionTree()],
            'rf': ['Random Forest', algs.RandomForest()],
            # 'ann': ['ANN', algs.ANN(epochs=5)],
        }

    def measureFitTime(self, alg, X, y, repeats=5):
        fastestRun = None

        for i in range(repeats):
            model = self.algorithms[alg][1]
            startTime = time.time()
            model.fit(X, y)
            duration = time.time() - startTime
            print(f'{alg} train, run {i+1}: {duration}s')

            if fastestRun is None or duration < fastestRun:
                fastestRun = duration

        return fastestRun

    def measurePredictTime(self, alg, trainX, trainY, X, y, repeats=5):
        fastestRun = None
        model = self.algorithms[alg][1]
        model.fit(trainX, trainY)

        for i in range(repeats):
            startTime = time.time()
            yPredicted = model.predict(X)
            duration = time.time() - startTime
            print(f'{alg} predict, run {i+1}: {duration}s')

            if fastestRun is None or duration < fastestRun:
                fastestRun = duration

        return fastestRun

    def readFile(self, path):
        data = pd.read_csv(f'{self.dirName}/{path}', index_col=0)
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
    pa = PerformanceAnalysis('../performance')

    # load testing set
    testX, testY = pa.readFile('data/AD_set_test.csv')

    # selected algs & sizes
    algs = ['logReg', 'svm', 'dt', 'rf', 'ann']
    datasetSizes = [20, 40, 60, 80, 100]

    fitTimes = {}
    predictTimes = {}

    for alg in algs:
        fitTimes[alg] = {}
        predictTimes[alg] = {}

        for size in datasetSizes:
            # load training set (testing set is always the same)
            trainX, trainY = pa.readFile(f'data/AD_set_train{size}_seed42.csv')

            # measure train time
            fitTimes[alg][size] = pa.measureFitTime(alg, trainX, trainY)

            # measure predict time
            predictTimes[alg][size] = pa.measurePredictTime(
                alg, trainX, trainY, testX, testY)

    fitTimesDF = pd.DataFrame(fitTimes)
    predictTimesDF = pd.DataFrame(predictTimes)

    # save times to .csv
    pa.saveDataToCSV(fitTimesDF, 'fit_time')
    pa.saveDataToCSV(predictTimesDF, 'predict_time')

    # plot results
    pa.savePlot(fitTimesDF.transpose(), 'fit_time')
    pa.savePlot(predictTimesDF.transpose(), 'predict_time')
    # pa.savePlot(fitTimesDF, 'fit_time')
    # pa.savePlot(predictTimesDF, 'predict_time')
