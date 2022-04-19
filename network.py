import random
import numpy as np
from Util.mnist_loader import editData, editExpandedData
import pickle
import gzip
import os.path

# class for the neural network
class network():
  """ 
  creates the network based of off the layer input
  layer input is a list with each element representing the amount of activators in that network
  an input of [10, 5, 1] will make a network with 3 layers with the first having 10 activators, next having 5, and last having 1
  savedNetwork will save the networks weights and biases to a file with the name you input so the network can be used in later executions
  """
  def __init__(self, layers, savedNetwork=None):
    self.savedNetwork = savedNetwork
    if savedNetwork and os.path.exists(f"Networks/{savedNetwork}.pkl.gz"):
      f = gzip.open(f"Networks/{savedNetwork}.pkl.gz", 'rb')
      self.layers, self.weights, self.biases = pickle.load(f, encoding='latin1')
      f.close()
      self.length = len(self.layers)
      self.saveNetwork(savedNetwork)
    else:
      self.layers = layers
      self.length = len(layers)
      self.biases = [np.random.randn(y, 1) for y in layers[1:]]
      self.weights = [np.random.randn(y, x)
                      for x, y in zip(layers[:-1], layers[1:])]

    self.function = sigmoid
    self.functionDerivative = sigmoidDerivative

  # runs the network and returns its predicted answer as a vector
  def forward(self, activation):
    for i in range(self.length-1):
      activation = self.function(np.dot(self.weights[i], activation) + self.biases[i])
    return activation

  # runs the network and returns its predicted answer as a number
  def runNetwork(self, input):
    output = list(self.forward(input))
    maxValue = max(output)
    maxIndex = output.index(maxValue)
    return maxIndex + 1

  # runs the network and returns true if its correct and false if incorrect
  def testNetwork(self, input, output):
    answer = self.runNetwork(input)
    if answer == list(output).index(1) + 1:
        return True
    else:
        return False

  # calculates the cost of a predicted answer verse the true answer
  def cost(self, input, output):
    cost = 0
    for i, o in zip(self.forward(input), output):
        cost += (i - o) ** 2
    return cost

  # uses given test data to return the accuracy of the network 
  def testAccuracy(self, testData):
    total = 0
    for x, y in testData:
      if self.testNetwork(x, y):
        total += 1
    return total / len(testData) * 100

  # uses given test data to return the average cost of the network
  def averageCost(self, testData):
    cost = 0
    for x, y in testData:
      cost += self.cost(x, y)
    return cost / len(testData)

  # seperates the training data into minibatches based on a given size
  def generateMiniBatches(self, trainingData, size):
    return [trainingData[x*size:(x+1)*size] for x in range(len(trainingData)//size)]

  """
  trains the network
  trainingData is a list of tuples each with the input and expected output
  learningRate affects how much each value is affected when it learns. Too high rates can lead to it overshooting the minimum and too low rates can lead it to get "stuck" in a local minimum
  miniBatchSize is the size of each minibatch
  cycles is the amount of times it iterates through the training data
  record will print how much it has progessed and the cost of the given iteration
  saveData will save the data to the network if it is true
  """
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
        if layer != 0: deltaB[layer-1] = np.array([da * dz for da, dz in zip(deltaA, self.functionDerivative(zs[layer-1]))]).reshape(len(zs[layer-1]),1)
    return deltaW, deltaB

  # derivative of the cost function to be used in calculating the gradient descent
  def costDerivative(self, x, y):
    return 2*(x-y)

  # saves the networks weights and biases to a file with the inputed name
  def saveNetwork(self, name):
    f = gzip.open(f'Networks/{name}.pkl.gz', 'w')
    pickle.dump((self.layers, self.weights, self.biases), f)
    f.close()
    print('Network saved')
    
# function that maps all real numbers to an interval of (0, 1)
def sigmoid(x):
    # sigmoid function
    return 1.0/(1.0+np.exp(-x))

# derivative of sigmoid function to be used in calculating the gradient descent
def sigmoidDerivative(x):
    # derivative of the sigmoid function
    return sigmoid(x)*(1-sigmoid(x))

# creates network
network1 = network([784, 30, 10])

# retrives data from mnist data
trainingData, testData = editExpandedData()

# gets the accuracy of before training
accuracy1 = network1.testAccuracy(testData)

# trains network
network1.train(trainingData, 3, 250, cycles=1, record=True, saveData=False)

# prints accuracy before and after training
print()
print()
print(accuracy1)
print(network1.testAccuracy(testData))