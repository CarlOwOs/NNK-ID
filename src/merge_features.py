import numpy as np
import faiss
from functools import partial
from scipy.spatial.distance import pdist, squareform
from queue import PriorityQueue

def calculate_rbf_similarity(input1, sigma=1):
    return np.exp(-squareform(pdist(input1, metric='sqeuclidean'))/(2*sigma))

# Merge the two closest points by KNN similarity
def get_merged_knn_two(features, distance, indices):
    n, dim = features.shape
    min_dist, idx_i, idx_j = 1e999, -1, -1

    for i in range(n):
        for j in range(np.shape(indices)[1]):
            if distance[i,j] < min_dist:
                min_dist = distance[i,j]
                idx_i, idx_j = i, indices[i,j]

    merge = (features[idx_i] + features[idx_j])/2
    new_features = np.delete(features, [idx_i, idx_j], axis=0) 
    new_features = np.vstack((new_features, merge))

    return new_features

# Merge the two closest points by NNK similarity
def get_merged_nnk_two(features, sim, indices):
    n, dim = features.shape
    max_sim, idx_i, idx_j = -1, -1, -1

    for i in range(n):
        for j in range(np.shape(indices)[1]):
            if sim[i,j] > max_sim:
                max_sim = sim[i,j]
                idx_i, idx_j = i, indices[i,j]

    merge = (features[idx_i] + features[idx_j])/2
    new_features = np.delete(features, [idx_i, idx_j], axis=0) 
    new_features = np.vstack((new_features, merge))

    return new_features

def get_merged_nnk_top_p(features, sim, indices, drop):
    n, dim = features.shape

    if n <= drop:
        return features

    q = PriorityQueue()

    for i in range(n):
        for j in range(np.shape(indices)[1]):
            q.put((-sim[i,j], (i,indices[i,j])))

    new_features = np.empty((0,np.shape(features)[1]), "float32")
    idx_visited = [False for i in range(n)]

    count = 0
    while count < drop and not q.empty():
        similarity__, (idx_i, idx_j) = q.get()

        if not idx_visited[idx_i] and not idx_visited[idx_j] and idx_i != idx_j:
            merge = (features[idx_i] + features[idx_j])/2
            new_features = np.vstack((new_features, merge))
            idx_visited[idx_i] = True
            idx_visited[idx_j] = True
            count += 1

    # add features of not visited points
    for i in range(n):
        if not idx_visited[i]:
            new_features = np.vstack((new_features, features[i]))
    
    return new_features

def get_merged_knn_top_p(features, distances, indices, drop):
    n, dim = features.shape

    if n <= drop:
        return features

    q = PriorityQueue()

    for i in range(n):
        for j in range(np.shape(indices)[1]):
            q.put((distances[i,j], (i,indices[i,j])))

    new_features = np.empty((0,np.shape(features)[1]), "float32")
    idx_visited = [False for i in range(n)]

    count = 0
    while count < drop and not q.empty():
        distance__, (idx_i, idx_j) = q.get()

        if not idx_visited[idx_i] and not idx_visited[idx_j] and idx_i != idx_j:
            merge = (features[idx_i] + features[idx_j])/2
            new_features = np.vstack((new_features, merge))
            idx_visited[idx_i] = True
            idx_visited[idx_j] = True
            count += 1

    # add features of not visited points
    for i in range(n):
        if not idx_visited[i]:
            new_features = np.vstack((new_features, features[i]))
    
    return new_features
