from model import model
from preprocessor import Preprocessor
import warnings

import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ingestion_program.data_manager import DataManager
from scoring_program.libscores import get_metric

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

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=67)

print("X train shape:", X_train.shape)
print("X test shape:", X_test.shape)
print("Y train shape:", Y_train.shape)
print("Y test shape:", Y_test.shape)

metric_name, scoring_function = get_metric()

M = model(RandomForestClassifier(n_estimators=310, bootstrap=False, warm_start=False), Preprocessor(show=True))
M.fit(X_train, Y_train)
aP = M.predict(X_test)
metric_name, scoring_function = get_metric()
print('Using scoring metric:', metric_name)

print(scoring_function(Y_test, aP))
