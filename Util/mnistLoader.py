import gzip
import pickle
from helperFunctions import vectorResult

def loadData():
    f = gzip.open('mnist.pkl.gz', 'rb')
    trainingData, validationData, testData = pickle.load(f, encoding='latin1')
    return (trainingData, validationData, testData)

def editData():
    td, vd, testd = loadData()
    trainingInputs = [x for x in td[0]]
    trainingResults = [vectorResult(y) for y in td[1]]
    trainingData = zip(trainingInputs, trainingData)
    return trainingData, vd, testd