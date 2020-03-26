import seaborn as sns
import warnings
from PIL import Image
from matplotlib import pyplot
import numpy as np
import random as rn
import path
from data_io import read_as_df
warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set()

# download data here : https://codalab.lri.fr/competitions/655#participate-get_starting_kit
data_dir = './public_data'
raw_data_dir = './public_data_raw'
data_name = 'plankton'


# The data are loaded as a Pandas Data Frame
data = read_as_df(data_dir + '/' + data_name)
rawData = read_as_df(raw_data_dir + '/' + data_name)


imgSampleData = rawData.iloc[15, :-1]
imgSampleData = np.array(imgSampleData, dtype=np.uint8)
imgSampleData.shape
imgSampleData = np.resize(imgSampleData, (100, 100))
print(imgSampleData.dtype)
print(imgSampleData.dtype)
print(imgSampleData.shape)

pyplot.imshow(imgSampleData).cmap
print(pyplot.imshow(imgSampleData).cmap)
pyplot.show()


def montreImage(index):
    imgSampleData = rawData.iloc[index, :-1]
    imgSampleData = np.array(imgSampleData, dtype=np.uint8)
    imgSampleData = np.resize(imgSampleData, (100, 100))
    pyplot.imshow(imgSampleData)
    pyplot.title(rawData.iloc[index, -1])
    pyplot.show()


def saveImage(index):
    imgSampleData = rawData.iloc[index, :-1]
    imgSampleData = np.array(imgSampleData, dtype=np.uint8)
    imgSampleData = np.resize(imgSampleData, (100, 100))
    img = Image.fromarray(imgSampleData, 'L')
    img.save("images / saved / {}.png".format(index))


def getImage(index):
    imgSampleData = rawData.iloc[index, :-1]
    imgSampleData = np.array(imgSampleData, dtype=np.uint8)
    imgSampleData = np.resize(imgSampleData, (100, 100))
    return imgSampleData


def binarizeImageArrayUsingMeans(img, means):
    res = np.array(img, dtype=bool)
    for x in range(100):
        for y in range(100):
            res[100 * y + x] = img[100 * y + x] > (means[100 + y] + means[x]) * 125
    return res


def binarizedImage_means(index):
    imgSampleData = np.array(rawData.iloc[index, :-1])
    imgInfos = np.array(data.iloc[index, :-4])

    binarizedImage = binarizeImageArrayUsingMeans(imgSampleData, imgInfos)
    binarizedImage = np.resize(binarizedImage, (100, 100))
    return binarizedImage


def derivatedImage(img):
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


def binarizedImageLocalDerivative(img):
    der = derivatedImage(img)
    quantile = np.quantile(der, 0.60)
    f = lambda x: 0 if x > quantile else 1
    return np.vectorize(f)(der)


def binarizedImage_localDerivative(index):
    imgSampleData = np.resize(np.array(rawData.iloc[index, :-1], dtype=np.uint8), (100, 100))
    # convertissement de l'array en image (matrice d'entiers)
    binarizedImage = binarizedImageLocalDerivative(imgSampleData)
    return binarizedImage


def extractPerimeter_withLocalDerivative(index):
    img = np.resize(np.array(rawData.iloc[index, :-1], dtype=np.uint8), (100, 100))
    der = derivatedImage(derivatedImage(img))
    quantile = np.quantile(der, 0.60)
    f = lambda x: 1 if x > quantile else 0
    pyplot.imshow(np.vectorize(f)(der))
    der = (np.vectorize(f)(der)).ravel()
    return sum(der) / len(der)


i = rn.choice(range(len(rawData)))
montreImage(i)
pyplot.imshow(binarizedImage_localDerivative(i))

extractPerimeter_withLocalDerivative(i)
