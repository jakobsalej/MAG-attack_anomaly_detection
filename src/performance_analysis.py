import time
from datetime import datetime
import os
import json
import argparse
import platform

from concurrent.futures import ThreadPoolExecutor
from memory_monitor import MemoryMonitor

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
        self.version = 1.1
        self.predictions = None
        self.verbose = verbose
        self.dirName = 'results/performance_analysis'
        self.resultsDirName = f'{self.dirName}/{resultsDir}'
        self.model = {}

        if not os.path.exists(self.dirName):
            os.mkdir(self.dirName)

        if not os.path.exists(self.resultsDirName):
            os.mkdir(self.resultsDirName)

    def measureFitTime(self, alg, algModel, X, y, repeats=5):
        times = []
        memory = []
        calibratedModel = None

        def run():
            model = None
            # Calibrated Classifier uses 5-fold CV by default
            model = CalibratedClassifierCV(base_estimator=algModel)
            # fit the model
            startTime = time.time()
            model.fit(X, y)
            duration = time.time() - startTime
            return duration, model


        for i in range(repeats):

            # Measure memory usage of each iteration
            with ThreadPoolExecutor() as executor:
                monitor = MemoryMonitor()
                memThread = executor.submit(monitor.measure_usage)
                try:
                    # Train the model
                    fnThread = executor.submit(run)
                    duration, calibratedModel = fnThread.result()

                    # Save run time
                    times.append(duration)
                finally:
                    # Save max memory usage
                    monitor.keep_measuring = False
                    maxUsage = memThread.result()
                    memory.append(maxUsage)

                    print(f'{alg} train, run {i+1}: {duration}s, memory usage: {maxUsage}')

        # Keep last trained model for predictions
        self.model[alg] = calibratedModel

        return times, memory

    def measurePredictTime(self, alg, X, y, repeats=5):
        times = []
        memory = []

        def run():
            startTime = time.time()
            _ = self.model[alg].predict(X)
            duration = time.time() - startTime
            return duration

        for i in range(repeats):

            # Measure memory usage of each iteration
            with ThreadPoolExecutor() as executor:
                monitor = MemoryMonitor()
                memThread = executor.submit(monitor.measure_usage)
                try:
                    # Predict
                    fnThread = executor.submit(run)
                    duration = fnThread.result()

                    # Save prediction time
                    times.append(duration)
                finally:
                    # Save max memory usage
                    monitor.keep_measuring = False
                    maxUsage = memThread.result()
                    memory.append(maxUsage)
                    
                    print(f'{alg} predict, run {i+1}: {duration}s, memory usage: {maxUsage}')
                        
        return times, memory

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
    parser = argparse.ArgumentParser()
    #parser.add_argument('-s','--size', type=float, nargs='+', default=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2])
    parser.add_argument('-s','--size', type=float, nargs='+', default=[0.01, 0.02, 0.05, 0.1, 0.15, 0.2])
    parser.add_argument('-a','--alg', type=str, nargs='+', default=['logReg', 'svm', 'dt', 'rf', 'ann'])
    parser.add_argument('-n','--name', type=str)
    args = parser.parse_args()

    # parameters
    datasetSizes = args.size
    algs = args.alg
    SHOULD_RESAMPLE = False
    RANDOM_SEED = 42
    PI = True

    # init
    folderName = f'{datetime.now().strftime("%d-%m-%Y(%H-%M-%S)")}_{args.name if args.name else ""}_{"all" if len(algs) == 5 else "_".join(algs)}_{"_".join(str(size) for size in datasetSizes)}'
    pa = PerformanceAnalysis(resultsDir=folderName)

    # save run settings
    settings = {
        'SCRIPT_VERSION': pa.version,
        'TRAINING_SET_SIZES': datasetSizes,
        'SELECTED_ALGORITHMS': algs,
        'PI_OPTIMIZED': PI,
        'RANDOM_SEED': RANDOM_SEED,
        'RESAMPLED_DATASET': SHOULD_RESAMPLE,
        'SYSTEM': platform.platform()
    }
    with open(f'{pa.resultsDirName}/run_settings.json', 'w') as fp:
        json.dump(settings, fp)

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
    averageFitMemory = {}
    averagePredictMemory = {}
    setSizes = {}

    for alg in algs:
        algTag = da.algorithms[alg][0]
        algModel = da.algorithms[alg][1]

        # Times
        averageFitTimes[alg] = {}
        averagePredictTimes[alg] = {}
        fitTimes = {}
        predictTimes = {}

        # Memory usage
        averageFitMemory[alg] = {}
        averagePredictMemory[alg] = {}
        fitMemory = {}
        predictMemory = {}

        for size in datasetSizes:
            # split training set further into smaller sets
            xTrainSmall, _, yTrainSmall, _ = da.splitTrainTest(xTrain, yTrain, trainSize=size, scale=False, resample=False, randomSeed=RANDOM_SEED)
            
            # TRAIN_SETS = {
            #     0.001: 'AD_subset_balanced_0.1.csv',
            #     0.002: 'AD_subset_balanced_0.2.csv',
            #     0.005: 'AD_subset_balanced_0.5.csv',
            #     0.01: 'AD_subset_balanced_1.csv',
            #     0.02: 'AD_subset_balanced_2.csv',
            #     0.05: 'AD_subset_balanced_5.csv',
            #     0.1: 'AD_subset_balanced_10.csv',
            #     0.15: 'AD_subset_balanced_15.csv',
            #     0.2: 'AD_subset_balanced_20.csv',
            # }
            # trainSetPath = f'data/AD_datoteke/C7_random/{TRAIN_SETS[size]}'

            # TRAIN_SETS = {
            #     0.001: 'AD_set_train_reduced_0.001_0.001.csv',
            #     0.002: 'AD_set_train_reduced_0.002_0.001.csv',
            #     0.005: 'AD_set_train_reduced_0.005_0.001.csv',
            #     0.01: 'AD_set_train_reduced_0.01_0.001.csv',
            #     0.02: 'AD_set_train_reduced_0.02_0.001.csv',
            #     0.05: 'AD_set_train_reduced_0.05_0.001.csv',
            #     0.1: 'AD_set_train_reduced_0.1_0.001.csv',
            #     0.15: 'AD_set_train_reduced_0.15_0.001.csv',
            #     0.2: 'AD_set_train_reduced_0.2_0.001.csv',
            # }
            # trainSetPath = f'data/AD_datoteke/Class_cluster/{TRAIN_SETS[size]}'

            # Read train set file
            # tmpData = pd.read_csv(trainSetPath)
            # xTrainSmall = tmpData.iloc[:,0:11]
            # yTrainSmall = tmpData.iloc[:,11] 
            
            # xTrainSmall = tmpData.iloc[:,1:12]
            # yTrainSmall = tmpData.iloc[:,12] 
            
            # print('DATA', tmpData)
            print('x train', xTrainSmall)
            print('y train',yTrainSmall)

            # save training/testing set size to file
            if size not in setSizes:
                setSizes[size] = {
                    'training': xTrainSmall.shape[0],
                    'testing': xTest.shape[0]
                }

            # measure train time, memory usage
            measuredFitTimes, measuredFitMemory = pa.measureFitTime(alg, algModel, xTrainSmall, yTrainSmall)
            fitTimes[size] = measuredFitTimes
            fitMemory[size] = measuredFitMemory
            averageFitTimes[alg][size] = np.average(measuredFitTimes)
            averageFitMemory[alg][size] = np.average(measuredFitMemory)

            # measure predict time
            measuredPredictTimes, measuredPredictMemory = pa.measurePredictTime(
                alg, xTest, yTest)
            predictTimes[size] = measuredPredictTimes
            predictMemory[size] = measuredPredictMemory
            averagePredictTimes[alg][size] = np.average(measuredPredictTimes)
            averagePredictMemory[alg][size] = np.average(measuredPredictMemory)

        # Save alg results to a file
        pa.saveDataToCSV(pd.DataFrame(fitTimes), f'fit_times_{algTag}')
        pa.saveDataToCSV(pd.DataFrame(predictTimes), f'predict_times_{algTag}')
        pa.saveDataToCSV(pd.DataFrame(fitMemory), f'fit_memory_usage_{algTag}')
        pa.saveDataToCSV(pd.DataFrame(predictMemory), f'predict_memory_usage_{algTag}')

    # save times, set sizes to .csv
    pa.saveDataToCSV(pd.DataFrame(averageFitTimes), 'avg_fit_times')
    pa.saveDataToCSV(pd.DataFrame(averagePredictTimes), 'avg_predict_times')
    pa.saveDataToCSV(pd.DataFrame(averageFitMemory), 'avg_fit_memory_usage')
    pa.saveDataToCSV(pd.DataFrame(averagePredictMemory), 'avg_predict_memory_usage')
    pa.saveDataToCSV(pd.DataFrame(setSizes), 'dataset_sizes')

