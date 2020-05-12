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
            # 'ann': ['ANN', algs.ANN(epochs=5)],
        }
