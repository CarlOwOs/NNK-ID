__author__ = "shekkizh"
"""Custom solver for solving non negative quadratic programs with positive definite matrices"""

import numpy as np
from scipy.linalg import get_lapack_funcs
import warnings
import faiss
from functools import partial
from scipy.spatial.distance import pdist, squareform
import statistics

def non_negative_qpsolver(A, b, x_init, x_tol, check_tol=-1, epsilon_low=-1, epsilon_high=-1):
    """
    Solves (1/2)x.T A x - b.T x
    :param x_init: Initial value for solution x
    :param x_tol: Smallest allowed non zero value for x_opt. Values below x_tol are made zero
    :param check_tol: Allowed tolerance for stopping criteria. If negative, uses x_tol value
    :param epsilon_high: maximum value of x during optimization
    :param epsilon_low: minimum value of x during optimization
    :return: x_opt, error
    """
    if epsilon_low < 0:
        epsilon_low = x_tol  # np.finfo(float).eps
    if epsilon_high < 0:
        epsilon_high = x_tol
    if check_tol < 0:
        check_tol = x_tol

    n = A.shape[0]
    # A = A + 1e-6 * np.eye(n)
    max_iter = 50 * n
    itr = 0
    # %%
    x_opt = np.reshape(x_init, (n, 1))

    # %%
    non_pruned_elements = x_opt > epsilon_low
    check = 1

    while (check > check_tol) and (itr < max_iter):
        x_opt_solver = np.zeros((n, 1))
        x_opt_solver[non_pruned_elements] = cholesky_solver(
            A[non_pruned_elements[:, 0], :][:, non_pruned_elements[:, 0]], b[non_pruned_elements[:, 0]], tol=x_tol)
        x_opt = x_opt_solver
        itr = itr + 1
        N = x_opt < epsilon_low
        if np.sum(N) > 0:
            check = np.max(np.abs(x_opt[N]))
        else:
            check = 0
        non_pruned_elements = np.logical_and(x_opt > epsilon_low, non_pruned_elements)

    x_opt[x_opt < x_tol] = 0
    return x_opt[:, 0]

def cholesky_solver(a, b, tol=1e-10, lower=False, overwrite_a=False, overwrite_b=False, clean=True):
    """Modified code from SciPy LinAlg routine"""

    a1 = np.atleast_2d(a)
    # Quick return for square empty array
    if a1.size == 0:
        return b

    potrf, = get_lapack_funcs(('potrf',), (a1,))
    c, info = potrf(a1, lower=lower, overwrite_a=overwrite_a, clean=clean)

    if info > 0:
        warnings.warn("Cholesky solver encountered positive semi-definite matrix -- possible duplicates in data")
        # return solve(a1, b, assume_a='sym', lower=lower, overwrite_a=overwrite_a, overwrite_b=overwrite_b,
        #              check_finite=False)
        c = c + tol*np.eye(b.size)

    potrs, = get_lapack_funcs(('potrs',), (c, b))
    x, info = potrs(c, b, lower=lower, overwrite_b=overwrite_b)
    return x

def calculate_ip_similarity(input1):
    return input1 @ input1.T ##

def calculate_rbf_similarity(input1, sigma):
    return np.exp(-squareform(pdist(input1, metric='sqeuclidean'))/(2*sigma)) #sqeuclidean -> euclidean

def get_nnk_weighted_graph(X, topk):
    features = X.detach().numpy()
    print(features.shape)
    n, dim = features.shape
    support_data = features

    index = faiss.IndexFlatL2(dim)

    # rbf
    similarity_func = partial(calculate_rbf_similarity)
    faiss_return_func = lambda x, sigma: np.exp(-x/(2*sigma))
    
    # ip
    # similarity_func = partial(calculate_ip_similarity) ##
    # faiss_return_func = lambda x: x ##

    index = faiss.index_cpu_to_all_gpus(index)
    index.add(support_data)
    similarities, indices = index.search(features, topk+1)
    
    similarities = similarities[:, 1:]
    indices = indices[:, 1:]

    #sigma = np.mean([np.quantile(similarities[i, :], 0.1) for i in range(n)])**2 
    sigma = (np.mean([similarities[i, 14] for i in range(n)])/3)**2

    for i, x in enumerate(features):
        neighbor_indices = indices[i, :]
        x_support = support_data[neighbor_indices]

        g_i = faiss_return_func(similarities[i], sigma) # Assuming images - dot products are always positive!! 
        G_i = similarity_func(input1=x_support, sigma=sigma)
        # g_i = x_support @ x ##
        # G_i = similarity_func(input1=x_support) ##
        x_opt = non_negative_qpsolver(G_i, g_i, g_i, x_tol=1e-10)
        similarities[i] = x_opt #/np.sum(x_opt) # We need the normalization to account for origin-shift invariance!            

    return similarities, indices
