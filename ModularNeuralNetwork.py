import numpy as np
from DataLoad import unpickle, load_data

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

print(list(set(train_y)))

# Flatten and normalize the data
train_x = train_x.reshape(train_x.shape[0], -1) / 255.0
test_x = test_x.reshape(test_x.shape[0], -1) / 255.0

# Convert into a one-hot vector (if the label value is 3, its one-hot vector will be [0,0,0,1,0,0,0,0,0,0]
train_y = to_categorical(train_y)
test_y = to_categorical(test_y)