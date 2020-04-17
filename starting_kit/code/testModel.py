from model import model
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

from preprocessor import Preprocessor
from ingestion_program.data_manager import DataManager
from sklearn.pipeline import Pipeline
from scoring_program.libscores import get_metric
from sklearn.base import BaseEstimator

warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set()

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # Note: if zDataManager is not ready, use the mother class DataManager

"""
We extract and split our sets of datas
"""
D = DataManager('plankton', './public_data', replace_missing=True)
X = D.data['X_train']
Y = D.data['Y_train']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

print("X train shape:", X_train.shape)
print("X test shape:", X_test.shape)
print("Y train shape:", Y_train.shape)
print("Y test shape:", Y_test.shape)

metric_name, scoring_function = get_metric()

a = model(RandomForestClassifier(n_estimators=310, bootstrap=False, warm_start=False))
a.fit(X_train, Y_train)
aP = a.predict(X_test)
metric_name, scoring_function = get_metric()
print('Using scoring metric:', metric_name)
#sc = make_scorer(scoring_function)
#a.printScore(sc,aP, Y_train)

#"""
#preprocessing fatas
#"""
#pre = Preprocessor()
#X_train, Y_train = pre.fit_transform(X_train, Y_train)
#X_test = pre.transform(X_test)

#print(X_train.shape, " ", Y_train.shape, " ", X_test.shape)

#model_list = [
#    Perceptron(),
#    GaussianNB(),
#    KNeighborsClassifier(),
#    DecisionTreeClassifier(),
#    RandomForestClassifier(),
#    AdaBoostClassifier(),
#    GradientBoostingClassifier()
#]

#param_list = [
#    {'tol': [1e-1, 1e-3, 1e-5], 'eta0': [0.01, 0.1, 1, 10]},
#    {'var_smoothing': [1e-1, 1e-5, 1e-9, 1e-14]},
#    {'n_neighbors': [1, 2, 3, 5, 10]},
#    {'max_depth': [10, 100, 500, 1000], 'max_features': ('sqrt', 'auto', 'log2')},
#    {'n_estimators': [290, 300, 310, 320, 330, 340], 'warm_start': (True, False), 'bootstrap': (True, False)},
#    {'n_estimators': [10, 50, 100, 150], 'learning_rate': [0.5, 0.75, 1, 1.25, 1.5]},
#    {'learning_rate': [0.05, 0.1, 0.2], 'n_estimators': [50, 100, 150, 200], 'warm_start': (True, False)}
#]

#"""
#Trouver le meilleur clasif __name__ ==sifieur avec les meilleurs paramètres
#"""
#clf = BestClf(model_list, param_list, X_train, Y_train)
#clf.train()

#print("meilleurs param = ", clf.bestParam)
#"""
#Le meilleur modèle est initialisé et on teste son score
#"""

#M = model(clf.bestClf, clf.bestParam)
#M.fit(X_train, Y_train)
#m_train = M.predict(X_train)



#print('Training score for the', metric_name, 'metric = %5.4f' % scoring_function(Y_test, m_train))
#scoresM = cross_val_score(M, X_train, Y_train.ravel(), cv=5, scoring=make_scorer(scoring_function), n_jobs=-1)
#print()

#"""
#Tests unitaires
#"""

#T = test(M.clf, M.param, X_train, Y_train, X_test, Y_test, scoring_function)
#T.allTests()
