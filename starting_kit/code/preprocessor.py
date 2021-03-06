"""
Created on Fri Mar 27 17:49:23 2020
@author: Jérôme, Pierre, George, Raphaël, Paul, Luqman
Last revised: Avr 04, 2020
Revision History :
    Avr 17, 2020 : Jérôme : moving tests to another file, add security to remove features only on X
    Avr 05, 2020 : Jérôme
    Avr 04, 2020 : Jérôme
    Mar 27, 2020 : Jérôme

This class aim to automate the preprocessing chain.
Briefly, we apply a series a algorithms in order to get data ready for the model.
First, we reduce the number of features in a clever way, we keep the features that are the most statistically likely to be useful.
After, we apply the PCA algorithm to reduce the number of features without losing data. It is used to reduce the calculation size.
The last step is the remove outliers, it is the data considered less usefull to analyse. Use perform a statistical test called chi2 to remove outliers.
"""

from sys import path
model_dir = 'sample_code_submission/'
result_dir = 'sample_result_submission/'
problem_dir = 'ingestion_program/'
score_dir = 'scoring_program/'
path.append(model_dir)
path.append(problem_dir)
path.append(score_dir)

import warnings
from data_manager import DataManager

from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
import seaborn as sns
import numpy as np
warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set()

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from sklearn.base import BaseEstimator


class Preprocessor(BaseEstimator):

    def __init__(self, show=False, pcaFeaturesNumber=40, PCA=False, Outliers=True, FeatureSelection=False):
        self.show = show
        self.fited = False
        self.pcaFeaturesNumber = pcaFeaturesNumber
        self.PCA = PCA
        self.Outliers = Outliers
        self.FeatureSelection = FeatureSelection

    def fit(self, X, Y):
        """ Learns from data, call fit methods of every aglorithm

        Parameters :
            X, Y the data
            pcaFeaturesNumber the final number of feature (optional, default = 70)

        Returns
            the class with everything fitted
        """

        if self.FeatureSelection:
            self.nbFeatures = self._featureSelectionFit(X, Y)
            self.feature_selection = SelectKBest(chi2, self.nbFeatures).fit(X, Y)
            X = self.feature_selection.transform(X)

        if self.PCA:
            self.pca = PCA(n_components=self.pcaFeaturesNumber).fit(X, Y)

        if self.Outliers:
            self.thresholdOutliers = self._removeOutliersFit(X)

        self.fited = True
        self.Xshape0 = X.shape[0]
        self.Xshape1 = X.shape[1]
        return self

    def fit_transform(self, X, Y):
        """ Learns from data, call fit methods of every aglorithm and transform the data

        Parameters :
            X, Y the data
            pcaFeaturesNumber the final number of feature (optional, default = 70)

        Returns
            X the data transformed
        """
        self.fited = True
        return self.fit(X, Y).transform(X, Y)

    def transform(self, X, Y=None):
        """ Transform the data from a previous learn

        Parameters :
            X, Y the data

        Returns
            X the data transformed
            Y the data transformed (optionnal)
        """
        if not self.fited:
            raise Exception("Cannot transform data that is not fit")
        else:
            if X.shape[1] == self.Xshape1 or Y is not None:  # X is a label tab
                if self.FeatureSelection:
                    X = self.feature_selection.transform(X)
                if self.PCA:
                    X = self.pca.transform(X)
            if X.shape[0] == self.Xshape0 or Y is not None:
                if self.Outliers:
                    X = self._removeOutliers(X)
            if Y is not None:
                if self.Outliers:
                    Y = self._removeOutliers(Y)
                return X, Y
            return X

    def _removeOutliersFit(self, X):
        """From X, _removeOutliersFit calculates the threshold to remove outliers

        Parameters :
            X the data

        Returns :
            the threshold for the outliers to be deleted
        """
        clf = LocalOutlierFactor()
        clf.fit_predict(X)
        arr = clf.negative_outlier_factor_.copy()
        self.arr = arr
        thresholds = np.flip(np.sort(arr))
        for diff in (max(arr) - min(arr)) / np.flip(np.arange(1, 4000, 100)):
            for i, th in enumerate(thresholds):
                if i > 10 and abs(thresholds[i] - thresholds[i - 1]) > diff:
                    if self.show:
                        print("prepro: threshold for outliers is {}".format(th))
                    return th
        return -1.7

    def _removeOutliers(self, X):
        """ Removes to outliers of X

        Parameters :
            X the data

        Returns :
            X without the outliers
        """
        threshold = self.thresholdOutliers
        arr = self.arr

        idxToDelete = []
        for i, d in enumerate(arr):
            if d < threshold:
                idxToDelete += [i]
        if self.show:
            print("prepro: ", len(idxToDelete), " data to delete")
        return np.delete(X, idxToDelete, axis=0)

    def _featureSelectionFit(self, X, Y):
        """ Finds the best number of features to keep

        Parameters :
            X, Y the data

        Returns :
            The number of features to keep
        """
        score, pvalue = chi2(X, Y)
        threshold = self._best_threshold_featureselect(pvalue, X, Y)

        nbFeatures = 0
        for i in pvalue:
            if(i < threshold):
                nbFeatures += 1

        if self.show:
            print("prepro: best number of features (with threshold = {}) is {}".format(threshold, nbFeatures))
        return nbFeatures

    def _best_threshold_featureselect(self, pvalue, x, y):
        """ Method used to help finding the best number of features to keep

        Parameters :
            pvalue the result of a chi2 test
            X, Y the data

        Returns :
            Threshold for selecting the features
        """
        thresholds = np.linspace(0, 1, 1000)
        res = np.zeros(len(thresholds))
        for i, threshold in enumerate(thresholds):
            k = 0
            for l in pvalue:
                if(l < threshold):
                    k += 1
            res[i] = k
            if i > 1 and res[i] - res[i - 1] < 1:
                return threshold
