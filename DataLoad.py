import numpy as np
import pickle

def unpickle(file):
    with open(file, 'rb') as byte_file:
        dict = pickle.load(byte_file, encoding='bytes')
    return dict

# Load all datasets and return the images and labels
def load_data():
    file_path = "./dataset/data_batch_"
    i = 1

    labels = []
    data = np.zeros((10000, 3072), dtype=np.uint8)
    index = 0
    while (i < 2):
        data_batch = unpickle(file_path + str(i))
        labels.extend(data_batch[b"labels"])
        data_values = data_batch[b"data"]
        
        for j in range(len(data_values)):
            data[index] = data_values[j]
            index = index + 1

        i = i + 1
    
    return [data, labels]