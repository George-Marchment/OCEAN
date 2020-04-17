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
from scoring_program.libscores import get_metric
from sklearn.pipeline import Pipeline

warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set()
from preprocessor import Preprocessor

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from sklearn.base import BaseEstimator


def tests():
    # more tests on our jupyter notebook, some functions here were tested on jupyter
    data_name = 'plankton'
    data_dir = './public_data'

    D = DataManager(data_name, data_dir, replace_missing=True)
    print("*** Original data ***")
    print(D)

    Prepro = Preprocessor()

    # Preprocess on the data and load it back into D
    #pp = Prepro.fit(D.data['X_train'], D.data['Y_train'])
    #X, Y = pp.transform(D.data['X_train'], D.data['Y_train'])
    #print(X.shape, Y.shape)
    D.data['X_train'] = Prepro.fit_transform(D.data['X_train'], D.data['Y_train'])
    D.data['Y_train'] = Prepro.transform(D.data['Y_train'])
    D.data['X_valid'] = Prepro.transform(D.data['X_valid'])
    D.data['X_test'] = Prepro.transform(D.data['X_test'])

    # Here show something that proves that the preprocessing worked fine
    print("*** Transformed data ***")
    print(D)


tests()
