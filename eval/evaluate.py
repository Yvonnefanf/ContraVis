from sklearn.manifold import trustworthiness 

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