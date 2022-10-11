import numpy as np

# Loads a dataset from a csv - check the kind of data and load accordingly
def load_data(path):
    dataset = np.loadtxt(path, delimiter=',')
    dataset = dataset.astype("float32")
    dataset = np.transpose(dataset) # CHECK IF NECESSARY
    dataset = dataset.copy(order='C')
    return dataset
