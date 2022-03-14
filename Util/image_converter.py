from PIL import Image
import numpy as np

def imageToArray(fileName):
  image = Image.open(fileName)
  if image.size != (28, 28):
    image = image.resize((28, 28))
  array = np.asarray(image)
  array = np.sum(array, axis=2).reshape(784, 1)
  array = (255 - (array/3)) * 1.0 / 255.0
  return array