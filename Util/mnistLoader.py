import gzip
import pickle
import numpy as np

def loadData():
    f = gzip.open('Data/mnist.pkl.gz', 'rb')
    trainingData, validationData, testData = pickle.load(f, encoding='latin1')
    return (trainingData, validationData, testData)

def editData():
    tr_d, va_d, te_d = loadData()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorResult(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_results = [vectorResult(y) for y in va_d[1]]
    validation_data = zip(validation_inputs, validation_results)
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_results = [vectorResult(y) for y in te_d[1]]
    test_data = zip(test_inputs, test_results)
    return (training_data, validation_data, test_data)

def loadExpandedData():
    f = gzip.open('Data/expanded_mnist.pkl.gz', 'rb')
    trainingData, validationData, testData = pickle.load(f, encoding='latin1')
    return (trainingData, validationData, testData)

def editExpandedData():
    tr_d, va_d, te_d = loadExpandedData()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorResult(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_results = [vectorResult(y) for y in va_d[1]]
    validation_data = zip(validation_inputs, validation_results)
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_results = [vectorResult(y) for y in te_d[1]]
    test_data = zip(test_inputs, test_results)
    return (training_data, validation_data, test_data)

def vectorResult(i):
    zeros = np.zeros((10, 1))
    zeros[i] = 1.0
    return zeros