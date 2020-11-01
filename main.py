import os
from NeuralNetwork.PhotoWork    import fragmentation


def start():
    print('Start program')
    for fileDelete in os.listdir('NewPhoto'):
        os.remove(f"NewPhoto/{fileDelete}")
    print(os.listdir('NewPhoto'))
    for photoName in os.listdir('TeachingPhoto'):
        print(f"{photoName}")
        fragmentation(photoName)

    return print('The END')


if __name__ == '__main__':
    start()
