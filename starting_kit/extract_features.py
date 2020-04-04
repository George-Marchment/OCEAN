# TODO : corriger les erreurs undefnied name 'rawData'

import warnings

import seaborn as sns
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import filters


from data_io import write_as_csv,read_as_df


warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set()

class Extract_features:
    featureThatCanBeExtracted = ['perimeter']
    
    def __init__(self, toExtractTo = "extracted_features/plankton", imagePath = "./public_data_raw/plankton", typeArray = ["train","test","valid"],toExtract =['perimeter']):
        self.imagePath = imagePath
        self.typeArray = typeArray
        self.extractedDf = pd.DataFrame()
        self.featuresToExtract = toExtract
        """
        for feature in self.featuresToExtract:
            if not feature in featureThatCanBeExtracted:
                raise Exception("features asked are not compatible")
        """
        self.imageDf = read_as_df(imagePath)

        
    
    
    def _extract_features(self)
        for featureName in self.featuresToExtract:
            if featureName == self.featureThatCanBeExtracted[0]:
                extractedDf['perimeter'] = pd.Series([self.extractPerimeterFromImage(get_image(i)) for i in range(len(imageDf))])
                
    def _store_features(self):
        for typeName in self.typeArray:
            write_as_csv(extractedDf,self.toExtractTo,type = typeName)
    
    def _extractPerimeterFromImage(img):
        img = filters.sobel(img)
        img = img.ravel()
        return img.sum()/(len(img))
    
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
        print(index)
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
