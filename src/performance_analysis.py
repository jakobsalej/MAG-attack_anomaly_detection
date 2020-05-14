import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from algorithms import Algorithms


class PerformanceAnalysis:
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

    def measureFitTime(self, alg, X, y, repeats=5):
        fastestRun = None

        for i in range(repeats):
            model = self.algorithms[alg][1]
            startTime = time.time()
            model.fit(X, y)
            duration = time.time() - startTime
            # print(duration)

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
            # print(duration)

            if fastestRun is None or duration < fastestRun:
                fastestRun = duration

        return fastestRun

    def readFile(self, path):
        data = pd.read_csv(f'{self.dirName}/{path}')
        X = data.drop('normality', axis=1)
        y = data['normality']
        return X, y

    def savePlot(self, data, title, fileName):
        sns.set()
        plt.figure()
        plot = sns.barplot(data=data).set_title(title)
        plot.get_figure().savefig(f'{self.dirName}/{fileName}.png')


if __name__ == '__main__':
    pa = PerformanceAnalysis('../performance')
    trainX, trainY = pa.readFile('AD_set_train.csv')
    testX, testY = pa.readFile('AD_set_train.csv')

    # selected algs
    algs = ['logReg', 'svm', 'dt', 'rf']
    fitTimes = {}
    predictTimes = {}

    for alg in algs:
        # measure train time
        fitTimes[alg] = pa.measureFitTime(alg, trainX, trainY)

        # measure predict time
        predictTimes[alg] = pa.measurePredictTime(
            alg, trainX, trainY, testX, testY)

    # plot results
    fitTimesDF = pd.Series(fitTimes).to_frame('Fit Times')
    predictTimesDF = pd.Series(predictTimes).to_frame('Predict Times')
    pa.savePlot(fitTimesDF.transpose(),
                'Time to fit on training data [s]', 'fit_time')
    pa.savePlot(predictTimesDF.transpose(),
                'Time to predict test data [s]', 'predict_time')
