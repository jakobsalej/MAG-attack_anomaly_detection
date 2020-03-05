from data_prediction import DataPrediction
from data_analysis import DataAnalysis

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# sns.set()

# np.random.seed(10)


if __name__ == '__main__':
    # prepare the data
    da = DataAnalysis('data/mainSimulationAccessTraces.csv')
    da.prepareData()
    # data.print()

    # Run predictions
    trainScores = {'Samples': []}
    testScores = {'Samples': []}

    for datasetSize in [0.2, 0.4, 0.6, 0.8, 1]:
        sampleData = da.returnData(datasetSize)
        dp = DataPrediction(sampleData)
        noOfSamples, predictions = dp.getScores()

        print(noOfSamples, predictions)

        # add no. of samples to dict for plotting
        trainScores['Samples'].append(noOfSamples)
        testScores['Samples'].append(noOfSamples)

        # add accuracies for each algorithm
        for algName in predictions:
            if algName in trainScores:
                trainScores[algName].append(predictions[algName][0])
                testScores[algName].append(predictions[algName][1])
            else:
                trainScores[algName] = [predictions[algName][0]]
                testScores[algName] = [predictions[algName][1]]

    trainDF = pd.DataFrame(data=trainScores)
    testDF = pd.DataFrame(data=testScores)

    trainDF = trainDF.melt(
        'Samples', var_name='Algorithm', value_name='Accuracy')
    testDF = testDF.melt(
        'Samples', var_name='Algorithm', value_name='Accuracy')

    sns.pointplot(x='Samples', y='Accuracy', hue='Algorithm',
                  data=trainDF, legend=True, legend_out=True).set_title('Training set')

    sns.pointplot(x='Samples', y='Accuracy', hue='Algorithm',
                  data=testDF, legend=True, legend_out=True).set_title('Testing set')

    plt.show()
