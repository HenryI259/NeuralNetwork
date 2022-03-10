import random
import numpy as np
from Util.mnistLoader import editData, editExpandedData
import pickle
import gzip
import os.path

class network():
  def __init__(self, layers, savedNetwork=None):
    self.savedNetwork = savedNetwork
    if savedNetwork and os.path.exists(f"Networks/{savedNetwork}.pkl.gz"):
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

  def runNetwork(self, input):
    output = list(self.forward(input))
    maxValue = max(output)
    maxIndex = output.index(maxValue)
    return maxIndex + 1

  def testNetwork(self, input, output):
    answer = self.runNetwork(input)
    if answer == list(output).index(1) + 1:
        return True
    else:
        return False

  def cost(self, input, output):
    cost = 0
    for i, o in zip(self.forward(input), output):
        cost += (i - o) ** 2
    return cost

  def testAccuracy(self, testData):
    total = 0
    for x, y in testData:
      if self.testNetwork(x, y):
        total += 1
    return total / len(testData) * 100

  def averageCost(self, testData):
    cost = 0
    for x, y in testData:
      cost += self.cost(x, y)
    return cost / len(testData)

  def generateMiniBatches(self, trainingData, size):
    return [trainingData[x*size:(x+1)*size] for x in range(len(trainingData)//size)]

  def train(self, trainingData, learningRate, miniBatchSize, cycles=1, record=False, saveData=False):
    for cycle in range(cycles):
      for m, minibatch in zip(range(len(trainingData)//miniBatchSize), self.generateMiniBatches(trainingData, miniBatchSize)):
        if m%5 == 0 and record:
          print("Data set " + str(m*miniBatchSize))
          print(f"cost: {self.cost(*minibatch[0])}")
          if self.testNetwork(*minibatch[0]):
            print("Correct")
          else:
            print("Incorrect")

        deltaW = [np.zeros(y*x).reshape(y, x)
                  for x, y in zip(self.layers[:-1], self.layers[1:])]
        deltaB = [np.zeros(y).reshape(y, 1) for y in self.layers[1:]]
        for x, y in minibatch:
          dw, db = self.backprop(x, y)
          deltaW = np.add(deltaW, dw)
          deltaB = np.add(deltaB, db)
        self.weights = [w-(nw*learningRate/miniBatchSize) for w, nw in zip(self.weights, deltaW)]
        self.biases = [b-(nb*learningRate/miniBatchSize) for b, nb in zip(self.biases, deltaB)]

        if m%5 == 0 and saveData and self.savedNetwork:
          self.saveNetwork(self.savedNetwork)
    if saveData and self.savedNetwork:
      self.saveNetwork(self.savedNetwork)

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
      deltaW[i] = np.dot(np.array([da * dz for da, dz in zip(deltaA[i+1], sigmoidDerivative(zs[i]))]).reshape(len(zs[i]),1), activations[i].reshape(1,len(activations[i])))
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
    print('Network saved')
    

def sigmoid(x):
    # sigmoid function
    return 1.0/(1.0+np.exp(-x))

def sigmoidDerivative(x):
    # derivative of the sigmoid function
    return sigmoid(x)*(1-sigmoid(x))

network1 = network([784, 16, 16, 10], 'networkTrial1')
trainingData, validationData, testData = editData()
td, vd, testd = editExpandedData()
network1.train(list(td)[38750:39000], 2, 50, cycles=1, record=True, saveData=True)

print(network1.testAccuracy(list(testData)))
#print(network1.averageCost(list(testData)))