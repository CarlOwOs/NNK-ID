import numpy as np
import faiss
import torch
from functools import partial
from scipy.spatial.distance import pdist, squareform
from queue import PriorityQueue

def get_merged_nnk_top_p(X, sim, indices, drop):

    features = X.detach().numpy()

    n, dim = features.shape

    if n <= drop:
        return X

    q = PriorityQueue()

    for i in range(n):
        for j in range(np.shape(indices)[1]):
            q.put((-sim[i,j], (i,indices[i,j])))
    
    new_features = np.empty((0,np.shape(features)[1]), "float32")
    idx_visited = [False for i in range(n)]

    count = 0
    while count < drop and not q.empty(): #drop
        similarity__, (idx_i, idx_j) = q.get()

        if not idx_visited[idx_i] and not idx_visited[idx_j]:
            merge_features = (features[idx_i] + features[idx_j])/2
            new_features = np.vstack((new_features, merge_features))
            idx_visited[idx_i] = True
            idx_visited[idx_j] = True
            count += 1

    # add not visited points
    for i in range(n):
        if not idx_visited[i]:
            new_features = np.vstack((new_features, features[i]))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return torch.from_numpy(new_features).to(torch.device(device))
