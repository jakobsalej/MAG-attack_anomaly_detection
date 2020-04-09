from keras.models import Sequential
from keras.layers import Dense, Activation


class ANN():
    def __init__(self):
        pass

    def getModel(self):
        model = Sequential()

        # hidden layer with 32 nodes
        model.add(Dense(32, input_dim=11, activation='relu'))

        # there are 8 output classes
        model.add(Dense(8, activation='softmax'))

        # compile model
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam', metrics=['sparse_categorical_accuracy', 'accuracy'])

        return model
