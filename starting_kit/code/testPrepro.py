from preprocessor import Preprocessor
import warnings

import seaborn as sns
from ingestion_program.data_manager import DataManager

warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set()

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)


def tests():
    # more tests on our jupyter notebook, some functions here were tested on jupyter
    data_name = 'plankton'
    data_dir = './public_data'

    D = DataManager(data_name, data_dir, replace_missing=True)
    print("*** Original data ***")
    print(D)

    Prepro = Preprocessor(show=True, PCA=True, FeatureSelection=True)

    # Preprocess on the data and load it back into D
    # pp = Prepro.fit(D.data['X_train'], D.data['Y_train'])
    # X, Y = pp.transform(D.data['X_train'], D.data['Y_train'])
    # print(X.shape, Y.shape)
    D.data['X_train'], D.data['Y_train'] = Prepro.fit_transform(D.data['X_train'], D.data['Y_train'])
    D.data['Y_train'] = Prepro.transform(D.data['Y_train'])
    D.data['X_valid'] = Prepro.transform(D.data['X_valid'])
    D.data['X_test'] = Prepro.transform(D.data['X_test'])

    # Here show something that proves that the preprocessing worked fine
    print("*** Transformed data ***")
    print(D)


tests()
