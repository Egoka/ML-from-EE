import os

from skimage import io

import NeuralNetwork.NeuralTeaching as ML


def trainingList(dirPath):
    for photoName in os.listdir(dirPath):
        photo = io.imread(f"./{dirPath}/{photoName}")
        photo = ML.compression(photo, 144, 144)
        photo = ML.splitIntoThreeChannels(photo)
        photo = ML.strideLayer(photo, scaleMAP)
        ML.subsampleLayer(photo)
        yield


def start():
    print('Start program')
    dirPath = 'NewPhoto'
    newPhoto = trainingList(dirPath)
    for i in range(len(os.listdir(dirPath))):
        next(newPhoto)
    return print('The END')


if __name__ == '__main__':
    start()