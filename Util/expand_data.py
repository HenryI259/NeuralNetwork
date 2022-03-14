import pickle
import gzip
import os.path
import random
import numpy as np
from mnist_loader import *

def expandData():
    if os.path.exists('Data/mnist_expanded.pkl.gz'):
        print('This data already exists')
    else:
        f = gzip.open("Data/mnist.pkl.gz", 'rb')
        trainingData, validationData, testData = pickle.load(f, encoding='latin1')
        f.close()
        
        counter = 0
        expandedDataPairs = []
        for input, output in zip(trainingData[0], trainingData[1]):
            expandedDataPairs.append((input, output))
            image = np.reshape(input, (-1, 28))
            counter += 1
            if counter % 1000 == 0:
                print("Expanding image " + str(counter))

            for d, axis, index_position, index in [
            (1,  0, "first", 0),
            (-1, 0, "first", 27),
            (1,  1, "last",  0),
            (-1, 1, "last",  27)]:
                new_img = np.roll(image, d, axis)
                if index_position == "first": 
                    new_img[index, :] = np.zeros(28)
                else: 
                    new_img[:, index] = np.zeros(28)
                    expandedDataPairs.append((np.reshape(new_img, 784), output))

        random.shuffle(expandedDataPairs)
        expandedTrainingData = [list(d) for d in zip(*expandedDataPairs)]
        print("Saving expanded data")
        f = gzip.open('Data/expanded_mnist.pkl.gz', 'w')
        pickle.dump((expandedTrainingData, validationData, testData), f)
        f.close()

expandData()