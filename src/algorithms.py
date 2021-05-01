from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

from keras.wrappers.scikit_learn import KerasClassifier
from ann import ANN


class Algorithms:
    def __init__(self, verbose=0, jobs=-1, pi=False):
        self.verbose = verbose
        self.jobs = 1 if pi else jobs
        self.pi = pi

    def logisticRegression(self):
        return LogisticRegression(verbose=self.verbose, class_weight='balanced', max_iter=10000, n_jobs=self.jobs)

    def SVM(self, linear=False):
        if linear:
            return LinearSVC(dual=False, class_weight='balanced', verbose=self.verbose)
        return SVC(class_weight='balanced', verbose=self.verbose)
        

    def DecisionTree(self):
        return DecisionTreeClassifier()

    def RandomForest(self, nEstimators=100):
        return RandomForestClassifier(n_estimators=nEstimators, n_jobs=self.jobs)

    def ANN(self, epochs=10):
        ann = ANN()
        estimator = KerasClassifier(
            build_fn=ann.getModel, epochs=epochs, verbose=self.verbose)
        return estimator
