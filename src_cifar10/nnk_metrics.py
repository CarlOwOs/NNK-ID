import numpy as np

# Returns the mean, sdev and a histogram (10, (0,k)) of the number of NNK neighbors
def get_neighbor_stats(sim, k):
    hist = np.histogram(np.count_nonzero(sim,1), bins=np.arange(0, k+2,1))
    mean = np.mean(np.count_nonzero(sim,1))
    std = np.std(np.count_nonzero(sim,1))
    
    return hist, mean, std

# Returns the mean, sdev and a histogram (10, (0,k)) of the ratio of NNK neighbors to KNN neighbors
def get_neighbor_ratio(sim):
    k = np.shape(sim)[1]
    hist = np.histogram(np.count_nonzero(sim,1)/k, range=(0,1))
    mean = np.mean(np.count_nonzero(sim,1)/k)
    std = np.std(np.count_nonzero(sim,1)/k)
    return hist, mean, std

# Returns a sorted list (size D) of the average and standard deviation for all NNK graphs of the D singular values
def get_sorted_ssvs(features, sim, indices):
    sorted_ssvs = np.zeros((0,np.shape(features)[1]))

    for i in range(len(features)):
        if np.count_nonzero(sim[i,]) > 0:
            neighbors = np.array([features[indices[i,j],] for j in range(np.shape(indices)[1]) if sim[i,j] != 0])
            X = np.vstack([neighbors, np.array(features[i,])])
            _, ssvs, _ = np.linalg.svd(np.transpose(X) @ X)
            sorted_ssvs = np.vstack([sorted_ssvs, np.sort(ssvs)[::-1]])
            
    if len(sorted_ssvs) > 0: 
        mean_sorted_ssvs = np.mean(sorted_ssvs, axis=0)
        std_sorted_ssvs = np.std(sorted_ssvs, axis=0)
    else: # case no node has neighbors
        mean_sorted_ssvs = np.zeros(np.shape(features)[1])
        std_sorted_ssvs = np.zeros(np.shape(features)[1])

    return mean_sorted_ssvs, std_sorted_ssvs

# Returns the diameter of each polytope as well as the mean and standard deviation of the diameters
def get_diameter_poly(features, sim, indices):
    n = len(features)

    diam_per_poly = np.zeros(n)
    for i in range(n):
        nnk_neighbors = indices[i, np.nonzero(sim[i])[0]]
        if len(nnk_neighbors) > 0:
            a = features[nnk_neighbors]
            b = a.reshape(a.shape[0], 1, a.shape[1])
            distances = np.sqrt(np.einsum('ijk, ijk->ij', a-b, a-b))
            
            diam_per_poly[i] = distances.max() + 1e-6

        else:
            diam_per_poly[i] = 0

    return diam_per_poly


