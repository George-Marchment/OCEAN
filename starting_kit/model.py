# what we aim : https://scikit-learn.org/stable/developers/develop.html#apis-of-scikit-learn-objects

"""
Created on Fri Mar 27 17:49:23 2020
@author: Jérôme, Pierre, George, Raphaël, Paul, Luqman
Last revised: April 4, 2020
Revision History :
   April 4: Raphaël and Paul
We clearly improve all our class,
We also add unit test
"""

import pickle
import warnings
from os.path import isfile

import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import preprocessor as prepro
from ingestion_program.data_manager import DataManager
from scoring_program.libscores import get_metric

warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set()

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from sklearn.base import BaseEstimator
    # Note: if zDataManager is not ready, use the mother class DataManager


# NEED SAVE & LOAD

class model(BaseEstimator):
    """
We create our model we supposed is the best one with a given classifier with its parameters
"""

    def __init__(self, clf, param):
        """
        Initialisation of the model
        @clf : the classifier to initialize
        @param : the parameters associated with the classifier
        """
        # self.clf = RandomForestClassifier(n_estimators=310, bootstrap=False, warm_start=False)
        self.clf = clf.set_params(**param)
        self.param = param
        self.show = False
        self.fited = False
        self.n_components = 70
        # self.transformer = [PCA(self.n_components)]

    def fit(self, X, y):
        """
        Learning from data
        @X : Our training set of datas
        @y : the labels of our training set
        """
        # TODO : determine best parameters (eg: threshold see below)
        if not self.fited:
            self.clf.fit(X, y)
            self.fited = True

    def predict(self, X):
        """
        Prediction of the datas with our trained model
        @X : the testing set predicted by our model
        """
        return self.clf.predict(X)

    def predictProba(self, X):
        """
        Same as predict but return the probability of being in a class
        @X : the testing set predicted by our model
        """
        return self.clf.predict_proba(X)

    def save(self, path="./"):
        """
        Saving the trained model
        @path : the path to save the model
        """
        file = open(path + '_model.pickle', "wb")
        pickle.dump(self, file)
        file.close()

    def load(self, path="./"):
        """
        Loading the trained model
        @path : the path to load the model
        """
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile, 'rb') as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)

        return self


class BestParam(BaseEstimator):
    """
    A class to fin the best hyperparameters of a given classifier with given datas
    """

    def __init__(self, clf, listParam, X_train, Y_train):
        """
        Initialiaze the classifier with  a training set of datas
        @clf : the classifier
        @listParam : a list of parameters. It has to be called like {'name of the parameter' : (list of different values), ...}
        @X_train : the training set
        @Y_train : labels of the training set
        """
        self.clf = clf
        self.listParam = listParam
        self.X_train = X_train
        self.Y_train = Y_train
        self.bestParam = None
        self.bestScore = None

    def train(self):
        """
        Use the gridSearchCV algorithm to train our classifier and find its best parameters
        """
        tmpclf = GridSearchCV(self.clf, self.listParam, scoring='balanced_accuracy', n_jobs=-1)
        tmpclf.fit(self.X_train, self.Y_train.ravel())
        # print(tmpclf.best_params_)
        self.bestParam = tmpclf.best_params_
        self.bestScore = tmpclf.best_score_


class BestClf(BaseEstimator):
    """
    Find the best model with best parameters in a list of classifiers with a list of different parameters
    """

    def __init__(self, listClf, listParam, X, Y):
        """
        Initialize ou lists of classifiers and parameters with our training set of datas
        @listClf : a list of classifiers
        @listParam : a list of parameters. It has to be called like {'name of the parameter' : (list of different values), ...}
        @X : the training set
        @Y : label of the training set
        """
        self.listClf = listClf
        self.listParam = listParam
        self.X = X
        self.Y = Y
        self.score = 0
        self.bestClf = None
        self.bestParam = None
        if len(self.listClf) != len(self.listParam):
            print("Erreur, la liste de classifieur n'a pas la meme taille que la liste de parametres")
            exit(0)

    def train(self):
        """
        Find the best model by comparing the different scores
        """
        for i in range(len(self.listClf)):
            tmp = BestParam(self.listClf[i], self.listParam[i], self.X, self.Y)
            tmp.train()
            if tmp.bestScore > self.score:
                self.bestClf = self.listClf[i]
                self.score = tmp.bestScore
                self.bestParam = tmp.bestParam


class test(BaseEstimator):
    """
    A class to make unit tests on our model
    """

    def __init__(self, clf, param, X_train, Y_train, X_test, Y_test, scoring_function):
        """
        Initialize the model with a training and a testing set. The scoring_function is really useful here
        @clf : the classifier you want to test
        @param : the hyerparameters of your model
        @X_train : the training set
        @Y_train : labels of the training set
        @X_test : the testing set
        @Y_test : labels of the testing set
        @scoring_function : a function to get the score of your model
        """
        self.clf = clf
        self.param = param
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.scoring_function = scoring_function

    def test1(self):
        """
        Testing the scoring function
        """
        return self.scoring_function(self.Y_test, self.Y_test) == 1

    def test2(self):
        """
        Testing if the parameters of our
        """
        f = model(self.clf, self.param)
        g = model(self.clf, self.param)

        f.fit(self.X_train, self.Y_train)
        g.fit(self.X_train, self.Y_train)

        f_train = f.predict(self.X_test)
        g_train = g.predict(self.X_test)

        return self.scoring_function(g_train, self.Y_test) > self.scoring_function(f_train, self.Y_test)

    def allTests(self):
        if self.test1():
            print("test1 bon")
        else:
            print("problème sur scoring_function")

        if self.test2():
            print("test2 bon")
        else:
            print("Problème sur classifieur ou ses paramètres")


if __name__ == "__main__":
    """
    We extract and split our sets of datas
    """
    D = DataManager('plankton', './public_data', replace_missing=True)
    X = D.data['X_train']
    Y = D.data['Y_train'].ravel()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    """
    preprocessing fatas
    """
    pre = prepro.Preprocessor()
    X_train = pre.fit_transform(X_train, Y_train)
    X_test = pre.transform(X_test)

    model_list = [
        Perceptron(),
        GaussianNB(),
        KNeighborsClassifier(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
        GradientBoostingClassifier()
    ]

    param_list = [
        {'tol': [1e-1, 1e-3, 1e-5], 'eta0': [0.01, 0.1, 1, 10]},
        {'var_smoothing': [1e-1, 1e-5, 1e-9, 1e-14]},
        {'n_neighbors': [1, 2, 3, 5, 10]},
        {'max_depth': [10, 100, 500, 1000], 'max_features': ('sqrt', 'auto', 'log2')},
        {'n_estimators': [290, 300, 310, 320, 330, 340], 'warm_start': (True, False), 'bootstrap': (True, False)},
        {'n_estimators': [10, 50, 100, 150], 'learning_rate': [0.5, 0.75, 1, 1.25, 1.5]},
        {'learning_rate': [0.05, 0.1, 0.2], 'n_estimators': [50, 100, 150, 200], 'warm_start': (True, False)}
    ]

    """
    Trouver le meilleur classifieur avec les meilleurs paramètres
    """
    clf = BestClf(model_list, param_list, X_train, Y_train)
    clf.train()

    print("meilleurs param = ", clf.bestParam )
    """
    Le meilleur modèle est initialisé et on teste son score
    """

    M = model(clf.bestClf, clf.bestParam)
    M.fit(X_train, Y_train)
    m_train = M.predict(X_train)

    metric_name, scoring_function = get_metric()

    print('Training score for the', metric_name, 'metric = %5.4f' % scoring_function(Y_test, m_train))
    scoresM = cross_val_score(M, X_train, Y_train.ravel(), cv=5, scoring=make_scorer(scoring_function), n_jobs=-1)
    print()

    """
    Tests unitaires
    """

    T = test(M.clf, M.param, X_train, Y_train, X_test, Y_test, scoring_function)
    T.allTests()
