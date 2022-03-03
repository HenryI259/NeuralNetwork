import pickle
import gzip
import os.path
import random
from helperFunctions import *

def expandData(amount=1):
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
            counter += 1
            if counter % 1000 == 0:
                print("Expanding image " + str(counter))

            inputReformat = []
            inputRow = []
            for i in input:
                inputRow.append(i)
                if len(inputRow) == 28:
                    inputReformat.append(inputRow)
                    inputRow = []
            for i in range(amount):
                for xy, shiftBy in [('x', 1), ('x', -1), ('y', 1), ('y', -1)]:
                    if xy == 'x':
                        newImage = []
                        for row in inputReformat:
                            newImage.append(shift(row, shiftBy * (i+1)))
                    elif xy == 'y':
                        newImage = shift(inputReformat, shiftBy * (i+1))

                    newData = []
                    for row in newImage:
                        for item in row:
                            newData.append(item)

                    expandedDataPairs.append((newData, output))

        random.shuffle(expandedDataPairs)
        expandedTrainingData = [list(d) for d in zip(*expandedDataPairs)]
        print("Saving expanded data")
        f = gzip.open('Data/expanded_mnist.pkl.gz', 'w')
        pickle.dump((expandedTrainingData, validationData, testData), f)
        f.close()

#expandData()
print('done')
f = gzip.open('Data/expanded_mnist.pkl.gz', 'rb')
trainingData, validationData, testData = pickle.load(f, encoding='latin1')
f.close()
print(len(trainingData))