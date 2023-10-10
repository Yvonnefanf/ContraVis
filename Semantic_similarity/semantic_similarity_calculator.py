from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import svd

class SemanticSimCalculatorAbstractClass(ABC):
    @abstractmethod
    def __init__(self, ref_data_provider, tar_data_provider) -> None:
        pass
    
    @abstractmethod
    def calculator(self, *args, **kwargs):
        # return the similarity of each pair
        pass

'''Base class for semantic similairty calculator'''
class SemanticSimCalculator(SemanticSimCalculatorAbstractClass):
    def __init__(self, ref_data_provider, tar_data_provider) -> None:
        """Init parameters for semantic similarity calculator

        Parameters
        ----------
        ref_data_provider: data.DataProvider
            reference data provider
        tar_data_provider: data.DataProvider
            target data provider
    
        """
        self.ref_data_provider = ref_data_provider
        self.tar_data_provider = tar_data_provider
    
    def orthogonal_mapping(data1, data2):
        """
        use Orthogonal Mapping, map data2 to data1's space。
    
        :param data1: numpy array, shape (n_samples, n_features)
        :param data2: numpy array, shape (n_samples, n_features)
        :return: data2_mapped: numpy array, mapped data2
        """
        # step1: Centralized data
        data1_centered = data1 - np.mean(data1, axis=0)
        data2_centered = data2 - np.mean(data2, axis=0)
        # step2: Calculate the cross-covariance matrix
        C = data2_centered.T @ data1_centered
        # step3: singular value decomposition
        U, _, Vt = svd(C)

        # step4: Compute orthogonal mapping matrix
        W = U @ Vt

        # step5: Apply mapping matrix
        data2_mapped = data2_centered @ W
    
        return data2_mapped
    


    