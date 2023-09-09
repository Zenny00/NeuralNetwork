import numpy as np
import pickle
import matplotlib.pyplot as plt

def unpickle(file):
    with open(file, 'rb') as byte_file:
        dict = pickle.load(byte_file, encoding='bytes')
    return dict


def load_data():
    file_path = "./dataset/data_batch_"
    i = 1

    labels = []
    data = np.zeros((50000, 3072))

    while (i < 6):
        data_batch = unpickle(file_path + str(i))
        labels.extend(data_batch[b"labels"])
        data_values = data_batch[b"data"]
        
        for j in range(len(data_values)):
            data[j] = data_values[j]

        i = i + 1

    return [data, labels]

data = load_data()

print(len(data[0]), len(data[1]))

data_batch_1 = unpickle("./dataset/data_batch_1")

x_train = data_batch_1[b'data']
print(x_train.shape)

meta_file = unpickle("./dataset/batches.meta")
print(meta_file[b'label_names'])

image = data_batch_1[b'data'][0]
image = image.reshape(3, 32, 32)
image = image.transpose(1, 2, 0)

label = data_batch_1[b'labels'][0]
print(image.shape)

x_train = x_train.reshape(len(x_train), 3, 32, 32)
x_train = x_train.transpose(0, 2, 3, 1)

label_name = meta_file[b'label_names']

# take the images data from batch data
images = data_batch_1[b'data']
# reshape and transpose the images
images = images.reshape(len(images),3,32,32).transpose(0,2,3,1)
# take labels of the images 
labels = data_batch_1[b'labels']
# label names of the images
label_names = meta_file[b'label_names']

# dispaly random images
# define row and column of figure
rows, columns = 5, 5
# take random image idex id
imageId = np.random.randint(0, len(images), rows * columns)
# take images for above random image ids
images = images[imageId]
# take labels for these images only
labels = [labels[i] for i in imageId]

# define figure
fig=plt.figure(figsize=(10, 10))
# visualize these random images
for i in range(1, columns*rows +1):
    fig.add_subplot(rows, columns, i)
    plt.imshow(images[i-1])
    plt.xticks([])
    plt.yticks([])
    plt.title("{}"
          .format(label_names[labels[i-1]]))
plt.show()
