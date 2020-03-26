from sys import path
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor

import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import time
from sklearn.tree import DecisionTreeClassifier
from model import model
from data_manager import DataManager
from libscores import get_metric

warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set()
model_dir = 'sample_code_submission/'
result_dir = 'sample_result_submission/'
problem_dir = 'ingestion_program/'
score_dir = 'scoring_program/'
path.append(model_dir)
path.append(problem_dir)
path.append(score_dir)
# let those libraries after path because libraries are inside the path
from data_io import read_as_df
from data_manager import DataManager

# The sample_data directory should contain only a very small subset of the data
data_dir = './public_data'
raw_data_dir = './public_data_raw'
data_name = 'plankton'

# The data are loaded as a Pandas Data Frame
data = read_as_df(data_dir + '/' + data_name)
# rawData = read_as_df(raw_data_dir + '/' + data_name)

D = DataManager(data_name, data_dir, replace_missing=True)
print(D)


def removeOutliners(D, threshold=-1.7,  show=True):
    X = D.data['X_train']
    clf = LocalOutlierFactor(n_neighbors=7)
    clf.fit_predict(X)
    if show:
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))

        print(clf.negative_outlier_factor_)
        print("max is ", max(clf.negative_outlier_factor_))
        print("min is ", min(clf.negative_outlier_factor_))

        ax.plot(clf.negative_outlier_factor_, 'b.', label="data")
        ax.plot(threshold*np.ones(X.shape[0]), 'r', label="threshold= {}".format(threshold))
        ax.legend()
        ax.set_title("Representation of all points with outliners (under the threshold)")
        ax.set_xlabel("data")
        ax.set_ylabel("outliners")
        plt.show(fig)

    # save all indexes where clf.negative_outlier_factor_ is under the threshold
    arr = clf.negative_outlier_factor_.copy()
    idxToDelete = []
    for i in range(0, len(arr)):
        if (arr[i]) < threshold:
            idxToDelete += [i]

    # delete the outliners on X and Y

    if show:
        print(D.data['X_train'].shape)
        print(D.data['Y_train'].shape)

    D.data['X_train'] = np.delete(D.data['X_train'], idxToDelete, axis=0)
    D.data['Y_train'] = np.delete(D.data['Y_train'], idxToDelete, axis=0)

    if show:
        print(D.data['X_train'].shape)
        print(D.data['Y_train'].shape)


def PCAlg(D, n=70, show=True):
    pca = PCA(n_components=n).fit(D.data['X_train'], D.data['Y_train'])

    D.data['X_train'] = pca.transform(D.data['X_train'])
    D.data['X_valid'] = pca.transform(D.data['X_valid'])
    D.data['X_test'] = pca.transform(D.data['X_test'])

    if show:
        print(D.data['X_train'].shape)
        print(D.data['X_valid'].shape)
        print(D.data['X_test'].shape)


def featureSelection(D, show=True):
    fig, ax = plt.subplots(2, 1, figsize=(20, 10))

    threshold = 0.008

    score, pvalue = chi2(D.data['X_train'], D.data['Y_train'])[0],  chi2(D.data['X_train'], D.data['Y_train'])[1]

    if show:
        ax[0].plot(pvalue, 'b.', label="p-values of each feature")
        ax[0].plot(threshold*np.ones(len(score)), 'r', label="threshold= {}".format(threshold))
        ax[1].plot(score, 'b.', label="chi2 statistics of each feature")

        ax[0].legend()
        ax[1].legend()
        ax[0].set_title("For each feature, the p-value is calculated")
        ax[1].set_title("For each feature, the score is calculated")
        ax[0].set_xlabel("features")
        ax[1].set_xlabel("features")
        ax[0].set_ylabel("p-value")
        ax[1].set_ylabel("score")
        plt.show(fig)

    k = 0
    for i in pvalue:
        if(i < threshold):
            k += 1

    print("Best number of features (with threshold = {}) is {}".format(threshold, k))

    feature_selection = SelectKBest(chi2, k).fit(D.data['X_train'], D.data['Y_train'])

    D.data['X_train'] = feature_selection.transform(D.data['X_train'])
    D.data['X_valid'] = feature_selection.transform(D.data['X_valid'])
    D.data['X_test'] = feature_selection.transform(D.data['X_test'])
    if show:
        print(D.data['X_train'].shape)
        print(D.data['X_valid'].shape)
        print(D.data['X_test'].shape)


def get_precision_and_time_for_various_threshold(visible, threVals):
    metric_name, scoring_function = get_metric()
    idx = -1
    res = [[i, 0, 0, 0]for i in threVals]
    D = DataManager(data_name, data_dir, replace_missing=True)
    basicX = pd.DataFrame(D.data['X_train'])

    Y = D.data['Y_train']
    for var in threVals:
        idx += 1
        sel = VarianceThreshold(threshold=(var))
        X = pd.DataFrame(data=sel.fit_transform(basicX))
        alreadyDone = False
        for i in res[:idx]:
            if X.shape[1] == i[3]:
                res[idx][:] = i[:]
                res[idx][0] = var
                alreadyDone = True
                if (visible):
                    print(i, "=", X.shape)
                continue
        res[idx][3] = X.shape[1]
        if alreadyDone:
            continue
        if (visible):
            print("number of features after varThreshold = %d" % X.shape[1])
        M = DecisionTreeClassifier(max_depth=10, max_features='sqrt', random_state=42)
        start = time.process_time()
        M.fit(X, Y)
        res[idx][2] = time.process_time() - start
        # result_name = result_dir + data_name
        Y_hat = M.predict(X)
        if (visible):
            print(Y, " ", Y_hat)
        res[idx][1] = scoring_function(Y, Y_hat)
    return res


def graph_threshold_changes(visible=False, thresholdValues=np.linspace(0.001, 0.05, 10), n=1):
    moy = np.array(get_precision_and_time_for_various_threshold(visible, thresholdValues))
    i = n-1
    while i > 0:
        toAdd = np.array(get_precision_and_time_for_various_threshold(visible, thresholdValues))
        moy += toAdd
        i -= 1
    moy /= n
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    ax[0][0].plot(moy[:, 0], moy[:, 1])
    ax[0][0].set_xlabel("variance threshold")
    ax[0][0].set_ylabel("roc score")
    ax[0][0].legend()
    ax[0][1].plot(moy[:, 0], moy[:, 2])
    ax[0][1].set_xlabel("variance threshold")
    ax[0][1].set_ylabel("time of execution")
    ax[0][1].legend()
    ax[1][0].plot(moy[:, 0], moy[:, 3])
    ax[1][0].set_xlabel("variance threshold")
    ax[1][0].set_ylabel("feature number")
    ax[1][0].legend()
    ax[1][1].plot(moy[:, 1], moy[:, 2])
    ax[1][1].set_xlabel("roc score")
    ax[1][1].set_ylabel("time of execution")
    ax[1][1].legend()
    fig.show()


# graph_threshold_changes(thresholdValues=np.linspace(0.0001, 0.076, 90), n=5)
featureSelection(D)
PCAlg(D)
removeOutliners(D)
