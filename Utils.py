'''
Copyright (C) 2019 May 26 By JSH all rights reserved
Written by Sanghyeon Jo <josanghyeokn@gmail.com>
'''

import cv2
import numpy as np

def MNIST_Download(download_path, one_hot = True):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets(download_path, one_hot=one_hot)
    return mnist

if __name__ == '__main__':
    mnist = MNIST_Download("../../../Dataset/MNIST/", one_hot = True)
    
    train_images = mnist.train.images
    train_labels = mnist.train.labels

    test_images = mnist.test.images
    test_labels = mnist.test.labels

    print(train_images.shape)
    print(train_labels.shape)
    print(test_images.shape)
    print(test_labels.shape)

    '''
    (55000, 784)
    (55000, 10)
    (10000, 784)
    (10000, 10)
    '''
    for image, label in zip(train_images, train_labels):
        print('{} -> {}'.format(label, np.argmax(label)))
        
        image = (image * 255).astype(np.uint8)
        image = image.reshape((28, 28, 1))

        cv2.imshow('show', image)
        cv2.waitKey(0)