import os

from skimage import io

import NeuralNetwork.NeuralTeaching as ML


def trainingList(dirPath):
    for photoName in os.listdir(dirPath):
        photo = io.imread(f"./{dirPath}/{photoName}")
        ML.compression(photo, 48, 48)
        photo = ML.splitIntoThreeChannels(photo)
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