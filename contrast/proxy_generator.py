from abc import ABC, abstractmethod
import numpy as np


class ProxyGeneratorAbstractClass(ABC):
    @abstractmethod
    def __init__(self, ref_data_provider, tar_data_provider) -> None:
        pass

    @abstractmethod
    def construct(self, *args, **kwargs):
        # return proxy_high_dimesional data
        pass


class ProxyGenerator(ProxyGeneratorAbstractClass):
    '''Generate Proxy
    '''

    def __init__(self, ref_data_provider,tar_data_provider) -> None:
        """init parameters for spatiak dege constructor

        Parameters
        ref_data_provider : data.DataProvider
            reference data provider
        tar_data_provider : data.DataProvider
            target data provider
        """
        self.ref_data_provider = ref_data_provider
        self.tar_data_provider = tar_data_provider

    
    
   
    

        