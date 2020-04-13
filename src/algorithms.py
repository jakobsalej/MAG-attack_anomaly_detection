from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from keras.wrappers.scikit_learn import KerasClassifier
from ann import ANN


class Algorithms:
    def __init__(self, verbose=0):
        self.verbose = verbose

    def logisticRegression(self):
        return LogisticRegression(verbose=self.verbose,  max_iter=100, n_jobs=-1)

    def SVM(self):
        return SVC(verbose=self.verbose)

    def DecisionTree(self):
        return DecisionTreeClassifier()

    def RandomForest(self, nEstimators=100):
        return RandomForestClassifier(n_estimators=nEstimators, n_jobs=-1)

    def ANN(self, epochs=10):
        ann = ANN()
        estimator = KerasClassifier(
            build_fn=ann.getModel, epochs=epochs, verbose=self.verbose)
        return estimator
