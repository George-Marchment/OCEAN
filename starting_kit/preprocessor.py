"""
Created on Fri Mar 27 17:49:23 2020
@author: Jérôme, Pierre, George, Raphaël, Paul, Luqman
Last revised: Avr 04, 2020
Revision History :
    Avr 05, 2020 : Jérôme
    Avr 04, 2020 : Jérôme
    Mar 27, 2020 : Jérôme

This class aim to automate the preprocessing chain.
Briefly, we apply a series a algorithms in order to get data ready for the model.
First, we reduce the number of features in a clever way, we keep the features that are the most statistically likely to be useful.
After, we apply the PCA algorithm to reduce the number of features without losing data. It is used to reduce the calculation size.
The last step is the remove outliers, it is the data considered less usefull to analyse. Use perform a statistical test called chi2 to remove outliers.
"""
import warnings
import paths
from data_manager import DataManager

from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
import seaborn as sns
import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set()
paths

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from sklearn.base import BaseEstimator


class Preprocessor(BaseEstimator):

    def __init__(self):
        self.show = False
        self.fited = False

    def fit(self, X, Y, pcaFeaturesNumber=70):
        """ Learns from data, call fit methods of every aglorithm

        Parameters :
            X, Y the data
            pcaFeaturesNumber the final number of feature (optional, default = 70)

        Returns
            the class with everything fitted
        """
        self.nbFeatures = self._featureSelectionFit(X, Y)
        self.feature_selection = SelectKBest(chi2, self.nbFeatures).fit(X, Y)
        X2 = self.feature_selection.transform(X)
        self.pca = PCA(n_components=pcaFeaturesNumber).fit(X2, Y)
        self.thresholdOutliners = self._removeOutlinersFit(X)
        self.fited = True
        return self

    def fit_transform(self, X, Y, pcaFeaturesNumber=70):
        """ Learns from data, call fit methods of every aglorithm and transform the data

        Parameters :
            X, Y the data
            pcaFeaturesNumber the final number of feature (optional, default = 70)

        Returns
            X the data transformed
        """
        self.fited = True
        return self.fit(X, Y, pcaFeaturesNumber).transform(X)

    def transform(self, X, Y=None):
        """ Transform the data from a previous learn

        Parameters :
            X, Y the data

        Returns
            X the data transformed
            Y the data transformed (optionnal)
        """
        if not self.fited:
            raise Exception("Cannot transform is data is not fit")
        else:
            X = self.feature_selection.transform(X)
            X = self.pca.transform(X)
            X = self._removeOutliners(X)
            if Y is not None:
                print("remove outliners for Y")
                Y = self._removeOutliners(Y)
                return X, Y
            return X

    def _removeOutlinersFit(self, X):
        """From X, _removeOutlinersFit calculates the threshold to remove outliers

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
                    print("threshold for outliners is {}".format(th))
                    return th
        return -1.7

    def _removeOutliners(self, X):
        """ Removes to outliers of X

        Parameters :
            X the data

        Returns :
            X without the outliers
        """
        threshold = self.thresholdOutliners
        arr = self.arr

        idxToDelete = []
        for i, d in enumerate(arr):
            if d < threshold:
                idxToDelete += [i]
        print(len(idxToDelete), " data to delete")
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

        print("best number of features (with threshold = {}) is {}".format(threshold, nbFeatures))
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


if __name__ == "__main__":
    data_name = 'plankton'
    data_dir = './public_data'

    basename = 'Iris'
    D = DataManager(data_name, data_dir, replace_missing=True)
    print("*** Original data ***")
    print(D)

    Prepro = Preprocessor()

    # Preprocess on the data and load it back into D
    pp = Prepro.fit(D.data['X_train'], D.data['Y_train'])
    X, Y = pp.transform(D.data['X_train'], D.data['Y_train'])
    print(X.shape, Y.shape)
    D.data['X_train'] = Prepro.fit_transform(D.data['X_train'], D.data['Y_train'])
    D.data['X_valid'] = Prepro.transform(D.data['X_valid'])
    D.data['X_test'] = Prepro.transform(D.data['X_test'])

    # Here show something that proves that the preprocessing worked fine
    print("*** Transformed data ***")
    print(D)
