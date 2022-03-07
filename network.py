import random
import numpy as np
from Util.mnistLoader import editData
import pickle
import gzip

class network():
  def __init__(self, layers, savedNetwork=None):
    if savedNetwork:
      f = gzip.open(f"Networks/{savedNetwork}.pkl.gz", 'rb')
      self.layers, self.weights, self.biases = pickle.load(f, encoding='latin1')
      f.close()
      self.length = len(self.layers)
    else:
      self.layers = layers
      self.length = len(layers)
      self.biases = [np.random.randn(y, 1) for y in layers[1:]]
      self.weights = [np.random.randn(y, x)
                      for x, y in zip(layers[:-1], layers[1:])]

  def forward(self, activation):
    for i in range(self.length-1):
      activation = sigmoid(np.dot(self.weights[i], activation) + self.biases[i])
    return activation

  def generateMiniBatches(self, trainingData, size):
    return [trainingData[x*size:(x+1)*size] for x in range(len(trainingData)//size)]

  def train(self, trainingData, learningRate, miniBatchSize, cycles=1, record=False):
    for cycle in range(cycles):
      for minibatch in self.generateMiniBatches(trainingData, miniBatchSize):
        deltaW = [np.zeros(y*x).reshape(y, x)
                  for x, y in zip(self.layers[:-1], self.layers[1:])]
        deltaB = [np.zeros(y).reshape(y, 1) for y in self.layers[1:]]
        for x, y in minibatch:
          dw, db = self.backprop(x, y)
          deltaW = deltaW + dw
          deltaB = deltaB + db
        self.weights = [w-(nw*learningRate/miniBatchSize) for w, nw in zip(self.weights, deltaW)]
        self.biases = [b-(nb*learningRate/miniBatchSize) for b, nb in zip(self.biases, deltaB)]
        if record:
          print("output:")
          print(self.forward(minibatch[0][0]))
          print("true output:")
          print(minibatch[0][1])

  def backprop(self, input, output):
    activations = [input]
    zs = []
    deltaA = [[] for x in range(self.length)]
    deltaW = [np.zeros(y*x).reshape(y, x)
              for x, y in zip(self.layers[:-1], self.layers[1:])]
    deltaB = [np.zeros(y).reshape(y, 1) for y in self.layers[1:]]
    for layer in range(self.length-1):
      z = np.dot(self.weights[layer], activations[layer]) + self.biases[layer]
      zs.append(z)
      activations.append(sigmoid(z))
    for i, actives in zip(range(len(activations)),activations[::-1]):
      layer = len(activations)-i-1
      for j, a in zip(range(len(actives)), actives):
        if i == 0:
          deltaA[layer].append(self.costDerivative(a, output[j]))
        else:
          da = 0
          for k, d in zip(range(len(deltaA[layer+1])), deltaA[layer+1]):
            da += self.weights[layer][k][j] * sigmoidDerivative(zs[layer][k]) * d
          deltaA[layer].append(da)
    for i, layer in zip(range(len(self.weights)), self.weights):
      for j, out in zip(range(len(layer)), layer):
        for k, inp in zip(range(len(out)), out):
          deltaW[i][j][k] = deltaA[i+1][j] * activations[i][k] * sigmoidDerivative(zs[i][j])
    for i, layer in zip(range(len(self.biases)), self.biases):
      for j, b in zip(range(len(layer)), layer):
        deltaB[i][j] = deltaA[i+1][j] * sigmoidDerivative(zs[i][j])
    return deltaW, deltaB

  def costDerivative(self, x, y):
    return 2*(x-y)

  def saveNetwork(self, name):
    f = gzip.open(f'Networks/{name}.pkl.gz', 'w')
    pickle.dump((self.layers, self.weights, self.biases), f)
    f.close()
    

def sigmoid(x):
    # sigmoid function
    return 1.0/(1.0+np.exp(-x))

def sigmoidDerivative(x):
    # derivative of the sigmoid function
    return sigmoid(x)*(1-sigmoid(x))

network1 = network([784, 16, 16, 10])#, 'network1')
trainingData, validationData, testData = editData()
network1.train(list(trainingData)[0:10], 3, 5, cycles=1, record=True)
network1.saveNetwork('network1')