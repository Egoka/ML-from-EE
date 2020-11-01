import os


def trainingList():
    for photoName in os.listdir('TeachingPhoto'):
        print(f"./TeachingPhoto/{photoName}")
        yield


def start():
    print('Start program')
    newPhoto = trainingList()
    for i in range(len(os.listdir('TeachingPhoto'))):
        next(newPhoto)
    return print('The END')


if __name__ == '__main__':
    start()
