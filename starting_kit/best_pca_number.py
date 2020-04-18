# code for mutliprecessing can be found here : https://sebastianraschka.com/Articles/2014_multiprocessing.html
from model import model
from preprocessor import Preprocessor
import warnings

import seaborn as sns
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
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

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=42)

print("X train shape:", X_train.shape)
print("X test shape:", X_test.shape)
print("Y train shape:", Y_train.shape)
print("Y test shape:", Y_test.shape)

metric_name, scoring_function = get_metric()

print('Using scoring metric:', metric_name)

pcaTab = np.arange(5, 200, 5)
nbExec = 5

# Define an output queue
output = mp.Queue()

# define a example function


def getScore(nbPCAFeat, pos, output):
    M = model(RandomForestClassifier(n_estimators=310, bootstrap=False, warm_start=False), prepro=Preprocessor(pcaFeaturesNumber=nbPCAFeat, show=False, PCA=True, Outliers=False, FeatureSelection=False))
    score = 0
    for i in range(nbExec):
        M.fit(X_train, Y_train)
        aP = M.predict(X_test)
        metric_name, scoring_function = get_metric()
        score += scoring_function(Y_test, aP)
    print(nbPCAFeat, ":", score / nbExec)
    output.put((pos, score / nbExec))


# Setup a list of processes that we want to run
processes = [mp.Process(target=getScore, args=(pca, pos, output)) for pos, pca in enumerate(pcaTab)]

# Run processes
for p in processes:
    p.start()

# Exit the completed processes
for p in processes:
    p.join()

# Get process results from the output queue
results = [output.get() for p in processes]

results.sort()
results = [r[1] for r in results]

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(pcaTab, results)
ax.set_xlabel("dimension of features")
ax.set_ylabel("score")
ax.set_title("Graph showing the score depending of the number of features, calculated with PCA")
plt.savefig("best_features_pca")
