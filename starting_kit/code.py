from sys import path
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
# import pandas as pd
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


featureSelection(D)
PCAlg(D)
removeOutliners(D)
