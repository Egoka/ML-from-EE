from skimage import io
import numpy as np
import math


def fragmentation(photoName):
    image = io.imread(f"TeachingPhoto/{photoName}")
    image = 255 - image
    number = 0
    row = 0
    while image.shape[1] != row:
        if np.sum(image[:, row]) >= 300:
            row += 1
        elif row > 15:
            number += 1
            symbol = image[:, :row]
            symbol = shear_small(symbol)
            symbol = add_edge(symbol)
            symbol = symbol.astype('uint8')
            fileName = f"NewPhoto/{photoName.strip('.png')}.{number}.png"
            io.imsave(fileName, symbol)
            #########################################
            image, row = image[:, row:], 0
        else:
            image, row = image[:, row:], 0
            image = np.delete(image, row, axis=1)


def shear(image):
    row = 0
    while image.shape[0] != row:
        if np.sum(image[row, :]) >= 500:
            row += 1
        elif row > 100:
            image = image[:row, :]
        else:
            image, row = image[row:, :], 0
            image = np.delete(image, row, axis=0)
    return image


def shear_small(image):
    row, high, depth, symbol = 0, 0, 0, False
    while image.shape[0] != row:
        if np.sum(image[row, :]) > 500:
            row += 1
            depth += 1
        elif depth > 0:
            if depth > 10:
                depth, high = 0, row
                symbol = True ^ symbol
            else:
                image, depth = np.delete(image, list(range(high, row)), axis=0), 0
                row -= high
        else:
            if not symbol:
                image = np.delete(image, row, axis=0)
            else:
                row += 1
                if image.shape[0] == row:
                    image = np.delete(image, list(range(high, row)), axis=0)
                    row = high
    return image


def add_edge(image):
    if image.shape[0] > image.shape[1]:
        factor = math.ceil(image.shape[0] / 28)
    else:
        factor = math.ceil(image.shape[1] / 28)
    factor = math.ceil(factor * 1.35)
    size = 28 * factor
    if image.shape[0] > image.shape[1]:
        additive_up = int(size / 2) - int(image.shape[0] / 2)
        additive_left = int(size / 2) - int(image.shape[1] / 2)
    else:
        additive_left = int(size / 2) - int(image.shape[1] / 2)
        additive_up = int(size / 2) - int(image.shape[0] / 2)
    image_and_edge = np.zeros((size, size), 'int')
    image_and_edge[additive_up:image.shape[0] + additive_up, additive_left:image.shape[1] + additive_left] = image
    image_and_edge = image_and_edge.astype('uint8')
    return image_and_edge


def print_photo(photo):
    io.imshow(photo)
    io.show()
