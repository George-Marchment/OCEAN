# what we aim : https://scikit-learn.org/stable/developers/develop.html#apis-of-scikit-learn-objects

"""
Created on Fri Mar 27 17:49:23 2020
@author: Jérôme, Pierre, George, Raphaël, Paul, Luqman
Last revised: April 23, 2020
Revision History :
   April 4: Raphaël and Paul
We clearly improve all our class,
We also add unit test

    April 23 : Paul
Remove a useless print (used as test)
Improve import


USAGE:
python preprocessing.py input_dir output_dir

input_dir: directory in which the data lie, in AutoML format
output_dir: results



"""

import pickle
import os
from os.path import isfile

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from preprocessor import Preprocessor
from sklearn.base import BaseEstimator

class model(BaseEstimator):
    """
We create our model we supposed is the best one with a given classifier with its parameters
"""

    def __init__(self, classifier=RandomForestClassifier(n_estimators=320, bootstrap=False, warm_start=True), prepro=Preprocessor()):
        """
        Initialisation of the model
        @clf : the classifier to initialize
        @param : the parameters associated with the classifier
        """
        self.clf = classifier
        self.is_trained = False
        self.n_components = 70
        self.prepro = prepro
        # self.pipe = Pipeline([('prepro',Preprocessor()),
        #                      ('clf',classifier)])
        # The pipeline above was not working so we removed it and used the prepro without pipeline

    def fit(self, X, Y):
        """
        Learning from data
        @X : Our training set of datas
        @Y : the labels of our training set
        """
        self.is_trained = False
        X, Y = self.prepro.fit_transform(X, Y)
        self.clf.fit(X, Y.ravel())
        self.is_trained = True

    def transform(self, X, Y):
        """
        Transforming data
        @X : Our training set of datas
        @y : the labels of our training set
        """
        X, Y = self.prepro.transform(X, Y)
        #print(" X : ", X.shape, " Y : ", Y.shape) -> We printed it because we had size problem while using prepro
        return X, Y

    def fit_transform(self, X, Y, nbPCAFeatures=None):
        """
        Learning and transform data
        @X : Our training set of datas
        @y : the labels of our training set
        """
        self.fit(X, Y)
        return self.transform(X, Y)

    def predict(self, X):
        """
        Prediction of the datas with our trained model
        @X : the testing set predicted by our model
        """
        if not self.is_trained:
            raise Exception("Data must be fit before performing classifier prediction")
        X = self.prepro.transform(X)
        return self.clf.predict(X)

    def predictProba(self, X):
        """
        Same as predict but return the probability of being in a class
        @X : the testing set predicted by our model
        """
        return self.clf.predict_proba(X)

    def printScore(self, scoringFunct, X, y):
        print(scoringFunct(X, y))

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

    def __init__(self, clf, listParam, X_train, Y_train, prepro = Preprocessor()):
        """
        Initialiaze the classifier with  a training set of datas
        @clf : the classifier
        @listParam : a list of parameters. It has to be called like {'name of the parameter' : (list of different values), ...}
        @X_train : the training set
        @Y_train : labels of the training set
        """
        self.clf = clf
        self.prepro = prepro
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
        a,b = self.prepro.fit_transform(self.X_train, self.Y_train)
        tmpclf.fit(a,b)
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


if __name__ == '__main__':


    if len(argv) == 1:  # Use the default input and output directories if no arguments are provided
        input_dir = "../public_data"
        output_dir = "../results"
    else:
        input_dir = argv[1]
        output_dir = argv[2]

    basename = 'plankton'

    '''
        D = DataManager(basename, input_dir, replace_missing=True)
    X = D.data['X_train']
    Y = D.data['Y_train'].ravel()



    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    metric_name, scoring_function = get_metric()

    pa = {'n_estimators': [310,315, 320,325, 330], 'warm_start': (True, False), 'bootstrap': (True, False), 'random_state' : [42]}
    #x2, y2 = Preprocessor.fit_transform(X_train, Y_train)
    B  = BestParam(RandomForestClassifier(), pa, X_train, Y_train)
    B.train()
    print("Meilleurs param :", B.bestParam)
    print("Meilleur score :" , B.bestScore)
    '''
