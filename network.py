import random
import numpy as np
from Util.mnist_loader import editData, editExpandedData
import pickle
import gzip
import os.path
from Util.image_converter import imageToArray

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

    self.function = sigmoid
    self.functionDerivative = sigmoidDerivative

  def forward(self, activation):
    for i in range(self.length-1):
      activation = self.function(np.dot(self.weights[i], activation) + self.biases[i])
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
      for m, minibatch in enumerate(self.generateMiniBatches(trainingData, miniBatchSize)):
        if m%5 == 0 and record:
          print("Cycle " + str(cycle))
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

        if m%50 == 0 and saveData and self.savedNetwork:
          self.saveNetwork(self.savedNetwork)
    if saveData and self.savedNetwork:
      self.saveNetwork(self.savedNetwork)

  def backprop(self, input, output):
    activations = [input]
    zs = []
    deltaW = [np.zeros(y*x).reshape(y, x)
              for x, y in zip(self.layers[:-1], self.layers[1:])]
    deltaB = [np.zeros(y).reshape(y, 1) for y in self.layers[1:]]
    for layer in range(self.length-1):
      z = np.dot(self.weights[layer], activations[layer]) + self.biases[layer]
      zs.append(z)
      activations.append(self.function(z))
    for i in range(len(activations)):
      layer = len(activations)-i-1
      if i == 0:
        a = activations[-1]
        deltaA = [self.costDerivative(a[x], output[x]) for x in range(len(a))]
        deltaB[layer-1] = np.array([da * dz for da, dz in zip(deltaA, self.functionDerivative(zs[-1]))]).reshape(len(zs[-1]),1)
      else:
        dzda = np.array([da * dz for da, dz in zip(deltaA, self.functionDerivative(zs[layer]))]).reshape(len(zs[layer]),1)
        deltaW[layer] = np.dot(dzda, activations[layer].reshape(1,len(activations[layer])))
        deltaA = np.dot(self.weights[layer].transpose(), dzda)
        deltaB[layer-1] = np.array([da * dz for da, dz in zip(deltaA, self.functionDerivative(zs[layer-1]))]).reshape(len(zs[layer-1]),1)
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


network1 = network([784, 30, 10], 'current_network')
#trainingData, validationData, testData = editData()
#td, vd, testd = editExpandedData()
#network1.train(list(td), 3, 250, cycles=1, record=True, saveData=True)

#print(network1.testAccuracy(list(testData)))
#print(network1.averageCost(list(testData)))

print(network1.forward(imageToArray('3.jpeg')))
print(network1.runNetwork(imageToArray('3.jpeg')))
#print(list(td)[0][0])
#print(imageToArray('4.jpeg'))