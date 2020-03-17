from data_prediction import DataPrediction
from data_analysis import DataAnalysis

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import scale, StandardScaler
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
import numpy as np
import pandas as pd


class ANN():
    def __init__(self, data, testSize=0.2, verbose=0):
        self.data = data
        self.predictions = None
        self.verbose = verbose

        # separate X and y
        X = data.drop('normality', axis=1)
        y = data['normality']

        # scale the X as part of preprocessing
        xScaled = StandardScaler().fit_transform(X)
        X = pd.DataFrame(xScaled, columns=X.columns)

        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(
            X, y, test_size=testSize, random_state=0)

        print('Number of samples:', self.xTrain.shape)
        print(self.xTrain)

    def getModel(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.xTrain.shape[1], activation='relu'))

        # there are 8 output classes
        model.add(Dense(8, activation='softmax'))

        # compile model
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam', metrics=['sparse_categorical_accuracy', 'accuracy'])

        return model

    def run(self):
        # train
        kfold = KFold(n_splits=5, shuffle=True)
        estimator = KerasClassifier(
            build_fn=self.getModel, epochs=10, verbose=1)
        #model.fit(self.xTrain, self.yTrain, epochs=10, validation_split=0.1)

        results = cross_val_score(
            estimator, self.xTrain, self.yTrain, cv=kfold)
        print('Results:', results.mean())


if __name__ == '__main__':
    # prepare the data
    da = DataAnalysis('mainSimulationAccessTraces.csv')
    da.prepareData()

    # Run predictions
    sampleData = da.returnData(0.2)

    ann = ANN(sampleData)
    ann.run()
