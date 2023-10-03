from sklearn.manifold import trustworthiness 
from sklearn.neighbors import NearestNeighbors
import numpy as np
from pynndescent import NNDescent

def evaluate_proj_nn_perseverance_trustworthiness(data, embedding, n_neighbors, metric="euclidean"):
    """
    evaluate projection function, nn preserving property using trustworthiness formula
    :param data: ndarray, high dimensional representations
    :param embedding: ndarray, low dimensional representations
    :param n_neighbors: int, the number of neighbors
    :param metric: str, by default "euclidean"
    :return nn property: float, nn preserving property
    """
    t = trustworthiness(data, embedding, n_neighbors=n_neighbors, metric=metric)
    return t

def evaluate_high_dimesion_trans_knn_preserving(data, data_transformed,k=15):
    """
    evaluate the high dimesnion transformation, nn preserving
    :param data: ndarray, high dimensional representations before transformation
    :param data_transformed: ndarray, high dimensional representations after transformation
    """
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(data)
    indices_original = nbrs.kneighbors(return_distance=False)
    
    nbrs_transformed = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(data_transformed)
    indices_transformed = nbrs_transformed.kneighbors(return_distance=False)
    
    # compare
    consistency_ratios = []
    for idx_orig, idx_trans in zip(indices_original, indices_transformed):
        intersection = len(np.intersect1d(idx_orig, idx_trans))
        consistency_ratio = intersection / k
        consistency_ratios.append(consistency_ratio)
        
    
    average_consistency_ratio = np.mean(consistency_ratios)
    
    return average_consistency_ratio

def evaluate_proj_nn_perseverance_knn(data, embedding, n_neighbors, metric="euclidean"):
    """
    evaluate projection function, nn preserving property using knn algorithm
    :param data: ndarray, high dimensional representations
    :param embedding: ndarray, low dimensional representations
    :param n_neighbors: int, the number of neighbors
    :param metric: str, by default "euclidean"
    :return nn property: float, nn preserving property
    """
    n_trees = 5 + int(round((data.shape[0]) ** 0.5 / 20.0))
    n_iters = max(5, int(round(np.log2(data.shape[0]))))
    # get nearest neighbors
    nnd = NNDescent(
        data,
        n_neighbors=n_neighbors,
        metric=metric,
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=60,
        verbose=True
    )
    high_ind, _ = nnd.neighbor_graph
    nnd = NNDescent(
        embedding,
        n_neighbors=n_neighbors,
        metric=metric,
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=60,
        verbose=True
    )
    low_ind, _ = nnd.neighbor_graph

    border_pres = np.zeros(len(data))
    for i in range(len(data)):
        border_pres[i] = len(np.intersect1d(high_ind[i], low_ind[i]))

    # return border_pres.mean(), border_pres.max(), border_pres.min()
    return border_pres.mean()


