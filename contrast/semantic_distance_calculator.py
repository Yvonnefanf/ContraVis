
import numpy as np

from abc import ABC, abstractmethod
import numpy as np
from sklearn.neighbors import NearestNeighbors

class SemanticDistacneCalculatorAbstractClass(ABC):
    @abstractmethod
    def __init__(self, ref_data_provider, tar_data_provider) -> None:
        pass
    
    # @abstractmethod
    # def calculator(self, *args, **kwargs):
    #     # return the similarity of each pair
    #     pass

'''Base class for semantic similairty calculator'''
class SemanticDistanceCalculator(SemanticDistacneCalculatorAbstractClass):
    def __init__(self, ref_data_provider, tar_data_provider, ref_data_, tar_data_, REF_EPOCH, TAR_EPOCH, n_neighbors=15) -> None:
        """Init parameters for semantic similarity calculator

        Parameters
        ----------
        ref_data_provider: data.DataProvider
            reference data provider
        tar_data_provider: data.DataProvider
            target data provider
    
        """

        self.n_neighbors = n_neighbors

        self.ref_data_provider = ref_data_provider
        self.tar_data_provider = tar_data_provider
        self.REF_EPOCH = REF_EPOCH
        self.TAR_EPOCH = TAR_EPOCH
        self.ref_data_ = ref_data_
        self.tar_data_ = tar_data_

        self.ref_train_data = ref_data_provider.train_representation(self.REF_EPOCH)
        self.tar_train_data = tar_data_provider.train_representation(self.TAR_EPOCH)
        self.ref_train_data = self.ref_train_data.reshape(len(self.ref_train_data), -1)
        self.tar_train_data = self.tar_train_data.reshape(len(self.tar_train_data), -1)

        #### get train data pred logit
        self.ref_pred = self.ref_data_provider.get_pred(self.REF_EPOCH, self.ref_train_data)
        self.tar_pred = self.tar_data_provider.get_pred(self.TAR_EPOCH, self.tar_train_data)

        #### get input data pred logit
        self.input_ref_pred = self.ref_data_provider.get_pred(self.REF_EPOCH, self.ref_data_)
        self.input_tar_pred = self.tar_data_provider.get_pred(self.TAR_EPOCH, self.tar_data_)

        #### get knn train data info for each input data
        self.ref_knn_dists, self.ref_knn_indices = self.k_nearest_neibour(self.ref_data_, self.ref_train_data)
        self.tar_knn_dists, self.tar_knn_indices = self.k_nearest_neibour(self.tar_data_, self.tar_train_data)

       
    # calculate the cosine similarity of 2 vectors
    def cosine_similarity(self, u, v):
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    
    def k_nearest_neibour(self, data, fitdata):
        print("start calculating the k nearest training data, neibour...")
        neigh = NearestNeighbors(n_neighbors=self.n_neighbors, radius=0.4)
        neigh.fit(fitdata)
        
        knn_dists, knn_indices = neigh.kneighbors(data, n_neighbors=self.n_neighbors, return_distance=True)
        return knn_dists, knn_indices
    
    def weighted_average(self, preds, dists):
        # change distance to weight
        weights = 1 / (dists + 1e-5)  # To prevent division by 0, add a small constant
        weights /= np.sum(weights)  # normalize

        # calculate the result
        weighted_preds = np.dot(weights, preds)
        return weighted_preds

    def tar_ref_train_data_semantic_similairty_(self,r_index,t_index):
        # get the prediction of that 2 samples
        pred_r = self.input_ref_pred[r_index]
        pred_t = self.input_tar_pred[t_index]
        # calculate the cos_similairty: 
        cos_similiarty = self.cosine_similarity(pred_r, pred_t)

        # calculate the r's k nearest training data neibour prediction
        ref_knn_dists= self.ref_knn_dists[r_index]
        ref_knn_indices = self.ref_knn_indices[r_index]
        ref_knn_pred = self.ref_pred[ref_knn_indices]


        # calculate the t's k nearest tarining data neibour prediction
        tar_knn_dists= self.tar_knn_dists[t_index]
        tar_knn_indices = self.tar_knn_indices[t_index]
        tar_knn_pred = self.tar_pred[tar_knn_indices]

        ref_weighted_pred = self.weighted_average(ref_knn_pred, ref_knn_dists)
        tar_weighted_pred = self.weighted_average(tar_knn_pred, tar_knn_dists)

        relative_cos_similarity = self.cosine_similarity(ref_weighted_pred, tar_weighted_pred)

        # print("relative_cos_similarity",relative_cos_similarity,"cos_similiarty",cos_similiarty )
        return relative_cos_similarity + cos_similiarty, cos_similiarty, relative_cos_similarity
    
    



        

        





        
    





    
