import gzip
import pickle
import numpy as np

# retrive mnist data from file and return all 3 sets of data
def loadData():
    f = gzip.open('Data/mnist.pkl.gz', 'rb')
    trainingData, validationData, testData = pickle.load(f, encoding='latin1')
    return (trainingData, validationData, testData)

# retrive data and edit it into data that is easier to use by the network
def editData():
    tr_d, va_d, te_d = loadData()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorResult(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_results = [vectorResult(y) for y in te_d[1]]
    test_data = zip(test_inputs, test_results)
    return (list(training_data), list(test_data))

# retrive expanded data from file and return all 3 sets of data
def loadExpandedData():
    f = gzip.open('Data/expanded_mnist.pkl.gz', 'rb')
    trainingData, validationData, testData = pickle.load(f, encoding='latin1')
    return (trainingData, validationData, testData)

# retrive expanded data and edit it into data that is easier to use by the network
def editExpandedData():
    tr_d, va_d, te_d = loadExpandedData()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorResult(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_results = [vectorResult(y) for y in te_d[1]]
    test_data = zip(test_inputs, test_results)
    return (list(training_data), list(test_data))

# converts a number into a 10-dimensional unit vector representing the input number
def vectorResult(i):
    zeros = np.zeros((10, 1))
    zeros[i] = 1.0
    return zeros