import os
import numpy as np
from skimage import io

import NeuralNetwork.NeuralTeaching as ML


def trainingList(dirPath):
    inputNeurons = (3 * 2 * 2) * 8 * 8
    numberOfEntities = 14
    numberOfHiddenNeurons = 350
    ###########################################################
    inputWeights = (np.random.rand(numberOfHiddenNeurons, inputNeurons) - 0.5)
    outputWeights = (np.random.rand(numberOfEntities, numberOfHiddenNeurons) - 0.5)
    scaleMAP_1_Layer = np.random.rand(6, 3, 3) - 0.5
    scaleMAP_2_Layer = np.random.rand(12, 3, 3) - 0.5
    scaleMAP_3_Layer = np.random.rand(12, 3, 3) - 0.5
    ###########################################################
    for photoName in os.listdir(dirPath):
        photo = io.imread(f"./{dirPath}/{photoName}")
        if len(photo.shape) <= 2: photo = photo.reshape(photo.shape[0], photo.shape[1], 1)
        photo = ML.compression(photo, 144, 144)
        photo = ML.splitIntoThreeChannels(photo)
        ###########################################################
        photo = ML.strideLayer(photo, scaleMAP_1_Layer, numberOfMap=2)
        photo = ML.subsampleLayer(photo)
        photo = ML.strideLayer(photo, scaleMAP_2_Layer, sizeArr=22, numberOfMap=2)
        photo = ML.subsampleLayer(photo)
        photo = ML.strideLayer(photo, scaleMAP_3_Layer, sizeArr=8)
        ###########################################################
        inputWeights, outputWeights = ML.neuralNetwork(photo, int(photoName[0]), inputWeights, outputWeights)
        yield
    ###########################################################
    np.save('Numpy_array/scaleMAP_1_Layer.npy', scaleMAP_1_Layer)
    np.save('Numpy_array/scaleMAP_2_Layer.npy', scaleMAP_2_Layer)
    np.save('Numpy_array/scaleMAP_3_Layer.npy', scaleMAP_3_Layer)
    np.save('Numpy_array/Weight_Table_INPUT.npy', inputWeights)
    np.save('Numpy_array/Weight_Table_OUTPUT.npy', outputWeights)
    yield


def start():
    print('Start program')
    dirPath = 'NewPhoto'
    newPhoto = trainingList(dirPath)
    for i in range(len(os.listdir(dirPath)) + 1):
        next(newPhoto)
    return print('The END')


if __name__ == '__main__':
    start()