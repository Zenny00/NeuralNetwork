import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Neural network core class
class NeuralNetwork:
    def __init__(self, input_neurons, hidden_neurons, output_neurons, learning_rate, num_epochs):
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.num_epochs = num_epochs

        # Here we simply initialize the weights and bias to random values
        # They will be updated through the training process

        # Link the weights from input layer to hidden layer
        self.wih = np.random.normal(0.0, pow(self.input_neurons, -0.5), (self.hidden_neurons, self.input_neurons))
        self.bih = 0

        # Link weights from hidden layer to output layer
        self.who = np.random.normal(0.0, pow(self.hidden_neurons, -0.5), (self.output_neurons, self.hidden_neurons))
        self.bho = 0

        self.lr = learning_rate

    def sigmoid_activation(self, z):
        # The return will be the sigmoid function given the value of z
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def sigmoid_activation_derivative(self, z):
        # The return will be the derivative of the sigmoid value of z
        return self.sigmoid_activation(z) * (1 - self.sigmoid_activation(z))
    
    # Forward propagation function
    def forward_propagation(self, input_list):
        inputs = np.array(input_list, ndmin=2).T # Transpose the matrix

        # Pass the input values into the hidden layer
        hidden_inputs = np.dot(self.wih, inputs) + self.bih

        # Get the output of the hidden layer 
        hidden_outputs = self.sigmoid_activation(hidden_inputs)

        # Pass the output from the hidden layer into the output layer
        final_inputs = np.dot(self.who, hidden_outputs) + self.bho

        # Get the output of the output layer
        yj = self.sigmoid_activation(final_inputs)

        return yj

    # Backpropagation function
    def backpropagation(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T # Transpose the matrix
        tj = np.array(targets_list, ndmin=2).T

        # Pass the input values into the hidden layer
        hidden_inputs = np.dot(self.wih, inputs) + self.bih

        # Get the output from the hidden layer
        hidden_outputs = self.sigmoid_activation(hidden_inputs)

        # Pass the outputs from the hidden layer into the output layer
        final_inputs = np.dot(self.who, hidden_outputs) + self.bho

        # Get output of the output layer
        yj = self.sigmoid_activation(final_inputs)

        # Get the error from the output layer
        output_errors = - (tj - yj) # The inverse difference between the expected and real outputs

        # Find error in the hidden layer
        hidden_errors = np.dot(self.who.T, output_errors)

        # Update all of the weights using Gradient Descent
        self.who -= self.lr * np.dot((output_errors * self.sigmoid_activation_derivative(yj)), np.transpose(hidden_outputs))
        self.wih -= self.lr * np.dot((hidden_errors * self.sigmoid_activation_derivative(hidden_outputs)), np.transpose(inputs))

        # Update bias values
        self.bho -= self.lr * (output_errors * self.sigmoid_activation_derivative(yj))
        self.bih -= self.lr * (hidden_errors * self.sigmoid_activation_derivative(hidden_outputs))
        pass

    # Performing gradient descent optimization using backpropagation
    def fit(self, inputs_list, targets_list):
        for epoch in range(self.num_epochs):
            self.backpropagation(inputs_list, targets_list)
            print(f"Epoch {epoch}/{self.num_epochs} completed.")

    def predict(self, X):
        outputs = self.forward_propagation(X).T
        return outputs

def unpickle(file):
    with open(file, 'rb') as byte_file:
        dict = pickle.load(byte_file, encoding='bytes')
    return dict

# Load all datasets and return the images and labels
def load_data():
    file_path = "./dataset/data_batch_"
    i = 1

    labels = []
    data = np.zeros((50000, 3072), dtype=np.uint8)
    index = 0
    while (i < 6):
        data_batch = unpickle(file_path + str(i))
        labels.extend(data_batch[b"labels"])
        data_values = data_batch[b"data"]
        
        for j in range(len(data_values)):
            data[index] = data_values[j]
            index = index + 1

        i = i + 1
    
    return [data, labels]

data = load_data()
images = data[0]
labels = data[1]

# Reshape and transpose the images
images = images.reshape(len(images), 3, 32, 32)
images = images.transpose(0, 2, 3, 1)

# Strings are stored as byte strings. Here a list comprehension is used to decode each string
label_names = [x.decode() for x in unpickle("./dataset/batches.meta")[b'label_names']]

# Get training and testing subsets of the dataset (45,000 training examples, 5,000 testing examples)
train_x = images[:45000]
train_y = labels[:45000]

test_x = images[45000:]
test_y = labels[45000:]

# Flatten and normalize the data
train_x = train_x.reshape(train_x.shape[0], -1) / 255.0
test_x = test_x.reshape(test_x.shape[0], -1) / 255.0

# Convert into a one-hot vector (if the label value is 3, its one-hot vector will be [0,0,0,1,0,0,0,0,0,0]
train_y = to_categorical(train_y)
test_y = to_categorical(test_y)

neural_network = NeuralNetwork(input_neurons=3072, hidden_neurons=128, output_neurons=10, learning_rate=0.01, num_epochs=1000)
neural_network.fit(inputs_list=train_x, targets_list=train_y)

# Predicting probabilities
probabilities = neural_network.predict(test_x)

# Convert into one-hot vector format
predictions = []

for probability in probabilities:
    max_index = np.argmax(probability)
    prediction = np.zeros_like(probability)
    prediction[max_index] = 1
    predictions.append(prediction)

print("Accuracy:",accuracy_score(predictions, y_test))
print("CR:", classification_report(predictions, y_test))

fig, axes = plt.subplots(2, 4,, figsize=(10,6))
for i, ax in enumerate(axes.flat):
    img_data = test_x[i].reshape((32, 32))
    ax.imshow(img_data)
    ax.set_xticks([])
    ax.set_yticks([])
    index = np.where(predictions[i] == 1)[0][0]
    label = label_names[index]
    true_label = label_names[np.argmax(test_y[i])]
    if label != true_label: # Incorrect prediction
        ax.set_xlabel(label, color='r')
    else:
        ax.set_xlabel(label)
plt.show()
