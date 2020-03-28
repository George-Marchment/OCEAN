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
from data_io import read_as_df
from sys import argv
from data_manager import DataManager

from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import VarianceThreshold
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.utils.estimator_checks import check_estimator

from libscores import get_metric
warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set()


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from sklearn.base import BaseEstimator
    # Note: if zDataManager is not ready, use the mother class DataManager


class Preprocessor(BaseEstimator):

    def __init__(self):
        self.show = False
        self.fited = False
        self.n_components = 70
        self.transformer = [PCA(self.n_components)]

    def extract_features(self):
        ...

    def fit(self, X, y=None):
        """
        Learning from data
        """
        # TODO : determine best parameters (eg: threshold see below)
        self.fited = True
        return self.transformer.fit(X, y)

    def fit_transform(self, X, y=None):
        # TODO : call fit then transform
        self.fited = True

        X = self.feature_selection.transform(X)
        return self.transformer.fit_transform(X)

    def transform(self, X, y=None):
        if not self.fited:
            raise Exception("Impossible to transform unfit data")
        else:
            return self.transformer.transform(X)

    def _removeOutliners(self, X):
        """
        clf = LocalOutlierFactor(n_neighbors=7)
        """
        self.threshold = -1.7
        clf = LocalOutlierFactor()
        clf.fit_predict(X)
        if self.show:
            fig, ax = plt.subplots(1, 1, figsize=(15, 15))

            print(clf.negative_outlier_factor_)
            print("max is ", max(clf.negative_outlier_factor_))
            print("min is ", min(clf.negative_outlier_factor_))

            ax.plot(clf.negative_outlier_factor_, 'b.', label="data")
            ax.plot(self.threshold * np.ones(X.shape[0]), 'r', label="self.threshold= {}".format(self.threshold))
            ax.legend()
            ax.set_title("Representation of all points with outliners (under the threshold)")
            ax.set_xlabel("data")
            ax.set_ylabel("outliners")
            plt.show(fig)

        # save all indexes where clf.negative_outlier_factor_ is under the threshold
        arr = clf.negative_outlier_factor_.copy()
        idxToDelete = []
        for i in range(0, len(arr)):
            if (arr[i]) < self.threshold:
                idxToDelete += [i]

        # delete the outliners on X and Y

        if self.show:
            print(D.data['X_train'].shape)
            print(D.data['Y_train'].shape)

        D.data['X_train'] = np.delete(D.data['X_train'], idxToDelete, axis=0)
        D.data['Y_train'] = np.delete(D.data['Y_train'], idxToDelete, axis=0)

        if self.show:
            print(D.data['X_train'].shape)
            print(D.data['Y_train'].shape)

    def _featureSelection(self, X, Y, threshold=0.008):

        score, pvalue = chi2(X, Y)

        if self.show:
            fig, ax = plt.subplots(2, 1, figsize=(20, 10))
            ax[0].plot(pvalue, 'b.', label="p-values of each feature")
            ax[0].plot(threshold * np.ones(len(score)), 'r', label="threshold= {}".format(threshold))
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

        self.feature_selection = SelectKBest(chi2, k).fit(X, Y)

        if self.show:
            print(D.data['X_train'].shape)
            print(D.data['X_valid'].shape)
            print(D.data['X_test'].shape)

    def _get_precision_and_time_for_various_threshold(visible, threVals):
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

    def _graph_threshold_changes(self, visible=False, thresholdValues=np.linspace(0.001, 0.05, 10), n=1):
        moy = np.array(self.get_precision_and_time_for_various_threshold(visible, thresholdValues))
        i = n - 1
        while i > 0:
            toAdd = np.array(self.get_precision_and_time_for_various_threshold(visible, thresholdValues))
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

    # TODO
    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"alpha": self.alpha, "recursive": self.recursive}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


if __name__ == "__main__":
    # We can use this to run this file as a script and test the Preprocessor
    check_estimator(Preprocessor)
    if len(argv) == 1:  # Use the default input and output directories if no arguments are provided
        input_dir = "../public_data_raw"
        output_dir = "../results"
    else:
        input_dir = argv[1]
        output_dir = argv[2]

    basename = 'Iris'
    D = DataManager(basename, input_dir)  # Load data
    print("*** Original data ***")
    print(D)

    Prepro = Preprocessor()

    # Preprocess on the data and load it back into D
    D.data['X_train'] = Prepro.fit_transform(D.data['X_train'], D.data['Y_train'])
    D.data['X_valid'] = Prepro.transform(D.data['X_valid'])
    D.data['X_test'] = Prepro.transform(D.data['X_test'])
    """
    D.feat_name = np.array(['PC1', 'PC2'])
    D.feat_type = np.array(['Numeric', 'Numeric'])
    """

    # Here show something that proves that the preprocessing worked fine
    print("*** Transformed data ***")
    print(D)


class ExtractFeatures():

    def _montreImage(self, index):
        imgSampleData = rawData.iloc[index, :-1]
        imgSampleData = np.array(imgSampleData, dtype=np.uint8)
        imgSampleData = np.resize(imgSampleData, (100, 100))
        plt.imshow(imgSampleData)
        plt.title(rawData.iloc[index, -1])
        plt.show()

    def _saveImage(index):
        imgSampleData = rawData.iloc[index, :-1]
        imgSampleData = np.array(imgSampleData, dtype=np.uint8)
        imgSampleData = np.resize(imgSampleData, (100, 100))
        img = Image.fromarray(imgSampleData, 'L')
        img.save("images / saved / {}.png".format(index))

    def _getImage(index):
        imgSampleData = rawData.iloc[index, :-1]
        imgSampleData = np.array(imgSampleData, dtype=np.uint8)
        imgSampleData = np.resize(imgSampleData, (100, 100))
        return imgSampleData

    def _binarizeImageArrayUsingMeans(img, means):
        res = np.array(img, dtype=bool)
        for x in range(100):
            for y in range(100):
                res[100 * y + x] = img[100 * y + x] > (means[100 + y] + means[x]) * 125
        return res

    def _binarizedImage_means(self, index):
        imgSampleData = np.array(rawData.iloc[index, :-1])
        imgInfos = np.array(data.iloc[index, :-4])

        binarizedImage = self.binarizeImageArrayUsingMeans(imgSampleData, imgInfos)
        binarizedImage = np.resize(binarizedImage, (100, 100))
        return binarizedImage

    def _derivatedImage(img):
        mean = sum(img.ravel()) * 0.000005  # moyenne  /  20
        imgTranspose = img.transpose()
        res = 0 * np.array(imgTranspose[1:-1, 1:-1], dtype=np.uint8)
        columnIdx = 0
        for column in imgTranspose[2:-2]:
            res[columnIdx] += np.uint8(mean * pow((column[2:] + column[:-2]) / column[1:-1], 1))
            columnIdx += 1
        res = res.transpose()
        lineIdx = 0
        for line in img[2:-2]:
            res[lineIdx] += np.uint8(mean * pow((line[2:] + line[:-2]) / line[1:-1], 1))
            lineIdx += 1
        return res

    def _binarizedImageLocalDerivative(self, img):
        der = self.derivatedImage(img)
        quantile = np.quantile(der, 0.60)
        f = lambda x: 0 if x > quantile else 1
        return np.vectorize(f)(der)

    def _binarizedImage_localDerivative(self, index):
        imgSampleData = np.resize(np.array(rawData.iloc[index, :-1], dtype=np.uint8), (100, 100))
        # convertissement de l'array en image (matrice d'entiers)
        binarizedImage = self.binarizedImageLocalDerivative(imgSampleData)
        return binarizedImage

    def extractPerimeter_withLocalDerivative(self, index):
        img = np.resize(np.array(rawData.iloc[index, :-1], dtype=np.uint8), (100, 100))
        der = self.derivatedImage(self.derivatedImage(img))
        quantile = np.quantile(der, 0.60)
        f = lambda x: 1 if x > quantile else 0
        plt.imshow(np.vectorize(f)(der))
        der = (np.vectorize(f)(der)).ravel()
        return sum(der) / len(der)
