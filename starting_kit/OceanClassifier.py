'''
Fichier créé le 29 Mars 2020
Dernière modification : 29 Mars 2020

Programme fait par Raphaël Bournet et Paul Michel dit Ferrer

Le but est de trouver le meilleur classifier et les meilleurs hyper-paramètres
pour cette competition :
https://codalab.lri.fr/competitions/623

'''

from sys import argv 
from sys import path

from sklearn.base import BaseEstimator
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from libscores import get_metric

from sklearn.metrics import balanced_accuracy_score as sklearn_metric
from sklearn.model_selection import StratifiedShuffleSplit


class classifier(BaseEstimator):
    ''' TODO''' 
    def __init__(self):
        self.clf = RandomForestClassifier(n_estimators= 310, bootstrap= False, warm_start= True)
        
    def fit(self, X, y):
        return self.clf.fit(X, y)
    
    def predict(self, X):
        return self.clf.predict(X)
    
    def predict_proba(self, X):
        return self.clf.predict_proba(X)
    

