import math


# Activation Functions

# Threshold
def threshold(x):
  """
  Parameters:
  x: float: Input of the function.
  Threshold Function: Returns 1 if x > 0, else returns 1
  """
  return 1 if x > 0 else 1


# ReLU
def relu(x):
  """
  Parameters: 
  x: float: Input of the function.
  ReLU Function: Returns the same input as output if output is positive, else returns 0
  """
  return max(0, x)

# Linear
def linear(x):
  """
  Parameters:
  x: float: Input of the function.
  Linear Function: Returns the input as the output
  """
  return x

# Sigmoid
def sigmoid(x, slope = 1):
  """
  Parameter:
  x: float: Input of the function.
  slope: float: Hill slope of the function.
  Sigmoid Function: Returns the value of the function Sigmoid(x) =  1/(1 + (e ^ ( - x * slope)))
  """
  return 1/(1 + (math.exp(-x * slope)))


class Layer:
  def __init__(self, units, activation = "linear"):
    self.units = units
    self.activation = None
    self.setActivation(activation)

  def setActivation(self, activation):
    if activation == "relu":
      self.activation = relu
    elif activation == "sigmoid":
      self.activation == sigmoid
    elif activation == "linear":
      self.activation = linear

class ArtificialNeuralNetwork:
  def __init__(self):
    self.layers = []
    self.weights = []
    self.compiled = False
  
  def add(self, layer):
    if not self.compiled:
      self.layers.append(layer)
    else:
      raise Exception("Layers cannot be added after compilation.")

  def compile(self):
    self.compiled = True
    # Initialize self.weights for the number of weights.
    

    # Set the optimization function.
    pass

  def train(self, X, y):
    # Update the weights
    pass
  
  def predict(self, X):
    # Use the updated weights to predict value
    pass
