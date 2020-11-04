import numpy as np


def compression(photo, heightOut, widthOut):
    heightRatio = int(photo.shape[0] / heightOut)
    widthRatio = int(photo.shape[1] / widthOut)
    outputPhoto = np.zeros((heightOut, widthOut), 'uint8')
    for i in range(0, heightOut):
        n = i * heightRatio
        for j in range(0, widthOut):
            h = j * widthRatio
            outputPhoto[i, j] = np.mean(photo[n:n + heightRatio, h:h + widthRatio])
    del photo, i, j, h, n, heightRatio, widthRatio, heightOut, widthOut
    return outputPhoto


def splitIntoThreeChannels(photo):
    if len(photo.shape) > 2:
        return np.array([photo[:, :, 0] / 255, photo[:, :, 1] / 255, photo[:, :, 2] / 255])
    else:
        return np.array([photo / 255, photo / 255, photo / 255])


def strideLayer(photo, scaleMAP, sizeArr=48, numberOfMap=1):
    if photo.shape[1] > photo.shape[2]:  # вертикальная
        step = round(photo.shape[2] / sizeArr)
        extraStep = round(photo.shape[1] / sizeArr)
    else:
        step = round(photo.shape[1] / sizeArr)
        extraStep = round(photo.shape[2] / sizeArr)
    outputArr, sumMap = np.zeros((3*numberOfMap, sizeArr, sizeArr), 'int'), 0
    for color in range(3*numberOfMap):
        for n in range(0, sizeArr * step, step):
            for h in range(0, sizeArr * extraStep, extraStep):
                for i in range(scaleMAP.shape[1]):
                    for j in range(scaleMAP.shape[2]):
                        try: sumMap += scaleMAP[numberOfMap-1, i, j] * photo[color, n + i, h + j]
                        except ValueError: continue
                outputArr[color, int(n / step), int(h / extraStep)], sumMap = sumMap, 0
    return outputArr


def subsampleLayer(photo, factor=2):
    outputArr = np.zeros((photo.shape[0], int(photo.shape[1] / factor), int(photo.shape[1] / factor)), 'int')
    for color in range(photo.shape[0]):
        for n in range(0, photo.shape[1], factor):
            for h in range(0, photo.shape[2], factor):
                outputArr[color, int(n / factor), int(h / factor)] = np.amax(photo[color, n:n + factor, h:h + factor])
    return outputArr


def neuralNetwork():
    pass
    # Цель слоя – классификация, моделирует сложную нелинейную функцию


def activationFunction():
    pass
    # В качестве функции активации в скрытых и выходном слоях применяется гиперболический тангенс


def backStrideLayer():
    pass
    # Вычисление δ происходит путем обратной свертки.
    # Для понимания обратно свертки, скользящее окно по карте признаков можно интерпретировать,
    # как обычный скрытый слой со связями между нейронами, но главное отличие — это то, что эти связи разделяемы,
    # то есть одна связь с конкретным значением веса может быть у нескольких пар нейронов, а не только одной.
    # Вычисление дельт происходит таким же образом, как и в скрытом слое полно связной сети.


def backSubsampleLayer():
    pass
    # Метод заключающийся в повороте ядра на 180 градусов и скользящем процессе сканирования сверхточной карты дельт
    # с измененными краевыми эффектами. Необходимо взять ядро сверхточной карты (следующего за под выборочным слоем)
    # повернуть его на 180 градусов и сделать обычную свертку по вычисленным ранее дельтам сверхточной карты,
    # но так чтобы окно сканирования выходило за пределы карты.

