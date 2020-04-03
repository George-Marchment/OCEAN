"""
Created on Fri Mar 27 17:49:23 2020
@author: Jérôme, Pierre, George, Raphaël, Paul, Luqman
Last revised: Mar 27, 2020
Revision History :
    Mar 27, 2020 : Jérôme

This class aim to automate the preprocessing chain.
Briefly, it will extract features from a set of data... TODO
"""

# what we aim : https://scikit-learn.org/stable/developers/develop.html#apis-of-scikit-learn-objects

import warnings
import path
from ingestion_program.data_io import read_as_df
from sys import argv
from ingestion_program.data_manager import DataManager
from sklearn.model_selection import train_test_split
import pickle

from sklearn.tree import DecisionTreeClassifier

from sklearn.base import BaseEstimator
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
from os.path import isfile
import seaborn as sns
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.utils.estimator_checks import check_estimator

from scoring_program.libscores import get_metric

warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set()

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from sklearn.base import BaseEstimator
    # Note: if zDataManager is not ready, use the mother class DataManager


# NEED SAVE & LOAD

class model(BaseEstimator):

    def __init__(self):
        self.clf = RandomForestClassifier(n_estimators=310, bootstrap=False, warm_start=False)
        self.show = False
        self.fited = False
        self.n_components = 70
        #self.transformer = [PCA(self.n_components)]

    def fit(self, X, y):
        """
        Learning from data
        """
        # TODO : determine best parameters (eg: threshold see below)
        if not self.fited:
            self.clf.fit(X, y)
            self.fited = True

    def predict(self, X):
        return self.clf.predict(X)

    def predictProba(self, X):
        return self.clf.predict_proba(X)

    def save(self, path="./"):
        file = open(path + '_model.pickle', "wb")
        pickle.dump(self, file)
        file.close()

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile, 'rb') as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)

        return self


class BestParam(BaseEstimator):

    def __init__(self, clf, listParam, X_train, Y_train):
        self.clf = clf
        self.listParam = listParam
        self.X_train = X_train
        self.Y_train = Y_train
        self.bestParam = None
        self.bestScore = None

    def train(self):
        tmpclf = GridSearchCV(self.clf, self.listParam, scoring='balanced_accuracy', n_jobs=-1)
        tmpclf.fit(self.X_train, self.Y_train.ravel())
        #print(tmpclf.best_params_)
        print("Score =", tmpclf.best_score_)
        self.bestParam = tmpclf.best_params_
        self.bestScore = tmpclf.best_score_

class BestClf(BaseEstimator):

    def __init__(self, listClf, listParam, X, Y):
        self.listClf = listClf
        self.listParam = listParam
        self.X = X
        self.Y = Y
        if len(self.listClf) != len(self.listParam):
            print("Erreur, la liste de classifieur n'a pas la meme taille que la liste de parametres")
            exit(0)


    def train(self):
        for i in range(len(self.listClf)):
            tmp = BestParam(self.listClf[i], self.listParam[i], X, Y)



if __name__ == "__main__":
    D = DataManager('plankton', './public_data', replace_missing=True)
    X = D.data['X_train']
    Y = D.data['Y_train']

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.33, random_state=42)
    #a = BestParam(Perceptron(),{'tol' : [1e-1, 1e-3, 1e-5], 'eta0' : [0.01, 0.1, 1, 10]},X_train, Y_train)
    #a.train()
    b = BestParam(GaussianNB(),{'var_smoothing' : [1e-1, 1e-5, 1e-9, 1e-14]}, X_train, Y_train)
    b.train()
    print(b.bestParam)