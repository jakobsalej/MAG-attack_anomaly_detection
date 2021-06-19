from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import os
import json
import argparse

from data_preparation import DataPreparation
from data_analysis import DataAnalysis
from performance_analysis import PerformanceAnalysis
from memory_monitor import MemoryMonitor

import numpy as np
import pandas as pd


def run():
    parser = argparse.ArgumentParser()
    #parser.add_argument('-s','--size', type=float, nargs='+', default=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2])
    # parser.add_argument('-s','--size', type=float, nargs='+', default=[0.01, 0.02, 0.05, 0.1, 0.15, 0.2])
    parser.add_argument('-s','--size', type=float, nargs='+', default=[0.01])
    # parser.add_argument('-a','--alg', type=str, nargs='+', default=['logReg', 'svm', 'svc', 'dt', 'rf', 'ann'])
    parser.add_argument('-a','--alg', type=str, nargs='+', default=['logReg'])
    args = parser.parse_args()
    
    pa = PerformanceAnalysis()

    # parameters
    datasetSizes = args.size
    algs = args.alg
    SHOULD_RESAMPLE = False
    RANDOM_SEED = 42
    PI = True

    # save run settings
    settings = {
        'TRAINING_SET_SIZES': datasetSizes,
        'SELECTED_ALGORITHMS': algs,
        'PI_OPTIMIZED': PI,
        'RANDOM_SEED': RANDOM_SEED,
        'RESAMPLED_DATASET': SHOULD_RESAMPLE
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
    setSizes = {}

    for alg in algs:
        algTag = da.algorithms[alg][0]
        algModel = da.algorithms[alg][1]

        averageFitTimes[alg] = {}
        averagePredictTimes[alg] = {}
        fitTimes = {}
        predictTimes = {}

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

            # measure train time
            measuredFitTimes = pa.measureFitTime(alg, algModel, xTrainSmall, yTrainSmall)
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

    # save times, set sizes to .csv
    pa.saveDataToCSV(pd.DataFrame(averageFitTimes), 'avg_fit_times')
    pa.saveDataToCSV(pd.DataFrame(averagePredictTimes), 'avg_predict_times')
    pa.saveDataToCSV(pd.DataFrame(setSizes), 'dataset_sizes')

with ThreadPoolExecutor() as executor:
    monitor = MemoryMonitor()
    mem_thread = executor.submit(monitor.measure_usage)
    try:
        fn_thread = executor.submit(run())
        result = fn_thread.result()
    finally:
        monitor.keep_measuring = False
        max_usage = mem_thread.result()
        
    print(f"Peak memory usage: {max_usage}")