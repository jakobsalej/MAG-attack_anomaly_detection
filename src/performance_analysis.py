import time
from datetime import datetime
import os

from data_preparation import DataPreparation
from data_analysis import DataAnalysis

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

    def measureFitTime(self, alg, X, y, repeats=10):
        times = []
        calibratedModel = None

        for i in range(repeats):
            calibratedModel = None
            # Calibrated Classifier uses 5-fold CV by default
            calibratedModel = CalibratedClassifierCV(
                base_estimator=da.algorithms[alg][1])

            # fit the model
            startTime = time.time()
            calibratedModel.fit(X, y)
            duration = time.time() - startTime
            print(f'{alg} train, run {i+1}: {duration}s')
            times.append(duration)

        # Keep last trained model for predictions
        self.model[alg] = calibratedModel

        return times

    def measurePredictTime(self, alg, X, y, repeats=10):
        times = []

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

    # variables
    SHOULD_RESAMPLE = False
    RANDOM_SEED = 42
    PI = False
    # algs = ['logReg', 'svm', 'dt', 'rf', 'ann']
    algs = ['dt']
    datasetSizes = [0.2, 0.4]

    # clean and preprocess data
    print('Preparing data...')
    dp = DataPreparation('data/mainSimulationAccessTraces.csv')
    dp.prepareData()
    sampleData = dp.returnData(1, randomSeed=RANDOM_SEED)

    # split data into X and y
    da = DataAnalysis(dirName=None, mode=1, pi=PI)
    X, y = da.splitXY(sampleData)

    # split data into training (80%) and testing (20%) set
    print('Creating training and testing dataset...')
    xTrain, xTest, yTrain, yTest = da.splitTrainTest(
        X, y, trainSize=0.8, scale=True, resample=SHOULD_RESAMPLE, randomSeed=RANDOM_SEED)

    # For charts
    averageFitTimes = {}
    averagePredictTimes = {}

    for alg in algs:
        algTag = da.algorithms[alg][0]

        averageFitTimes[alg] = {}
        averagePredictTimes[alg] = {}
        fitTimes = {}
        predictTimes = {}

        for size in datasetSizes:
            # split training set further into smaller sets]
            xTrainSmall, _, yTrainSmall, _ = da.splitTrainTest(
                X, y, trainSize=size, scale=False, resample=False, randomSeed=RANDOM_SEED)

            # measure train time
            measuredFitTimes = pa.measureFitTime(alg, xTrainSmall, yTrainSmall)
            fitTimes[size] = measuredFitTimes
            averageFitTimes[alg][size] = np.average(measuredFitTimes)

            # measure predict time
            measuredPredictTimes = pa.measurePredictTime(
                alg, xTest, yTest)
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
