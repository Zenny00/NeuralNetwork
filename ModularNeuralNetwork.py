import numpy as np
from DataLoad import unpickle, load_data
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        
        # Given an input X (usually a matrix) this function will compute the output Y (also a matrix)
        def forward_propagation(self, input):
            raise NotImplementedError
        
        # Compute the derivative of E with respect to X (dE/dX) for a given dE/dY. Update and parameters if needed
        def back_propagation(self, output_error, learning_rate):
            raise NotImplementedError

class FCLayer(Layer):
    # input_size is the number of input neurons
    # output_size is the number of output neurons
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5
        
    # Return the output of a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output
    
    # Computes dE/dW, dE/dB for a given output error (dE/dY). This returns the input error (dE/dX)
    def back_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error
    
class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
        
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output
    
    def back_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error
    
# Activation functions
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

# Loss functions
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size
    
data = load_data()
images = data[0]
labels = data[1]

# Reshape and transpose the images
images = images.reshape(len(images), 3, 32, 32)
images = images.transpose(0, 2, 3, 1)

# Strings are stored as byte strings. Here a list comprehension is used to decode each string
label_names = [x.decode() for x in unpickle("./dataset/batches.meta")[b'label_names']]

# Shuffle the dataset
# images, labels = shuffle(images, labels, random_state=True)

# Get training and testing subsets of the dataset (48,000 training examples, 48,000 testing examples)
train_x = images[:5000]
train_y = labels[:5000]

test_x = images[5000:]
test_y = labels[5000:]

fig, axes = plt.subplots(2, 4, figsize=(10,6))
for i, ax in enumerate(axes.flat):
    img_data = train_x[i].reshape((32, 32, 3))
    ax.imshow(img_data)
    ax.set_xticks([])
    ax.set_yticks([])
    label = label_names[train_y[i]]
    ax.set_xlabel(label)
plt.show()

# Flatten and normalize the data
train_x = train_x.reshape(train_x.shape[0], -1) / 255.0
test_x = test_x.reshape(test_x.shape[0], -1) / 255.0

# Convert into a one-hot vector (if the label value is 3, its one-hot vector will be [0,0,0,1,0,0,0,0,0,0]
train_y = to_categorical(train_y)
test_y = to_categorical(test_y)