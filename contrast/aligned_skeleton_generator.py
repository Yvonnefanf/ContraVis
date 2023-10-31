from abc import ABC, abstractmethod
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import sys
sys.path.append('..')
from contrast.aligned_skeleton_generator import SemanticDistanceCalculator
from sklearn.cluster import DBSCAN

class AlignedSkeletonGeneratorAbstractClass(ABC):
    @abstractmethod
    def __init__(self, ref_data_provider, tar_data_provider) -> None:
        pass


"""
For given reference and target high dimenisonal representations, generate aligned skeletons
The item in skeleton 

"""

class AlignedSkeletonGenerator(AlignedSkeletonGeneratorAbstractClass):
    def __init__(self, ref_data_provider, tar_data_provider, REF_EPOCH, TAR_EPOCH,max_depth=10,min_cluster_size=800) -> None:
        self.ref_data_provider = ref_data_provider
        self.tar_data_provider = tar_data_provider
        self.REF_EPOCH = REF_EPOCH
        self.TAR_EPOCH = TAR_EPOCH
        self.ref_data = self.ref_data_provider.train_representation(self.REF_EPOCH)
        self.ref_data = self.ref_data.reshape(self.ref_data.shape[0], self.ref_data.shape[1])
        self.tar_data = self.tar_data_provider.train_representation(self.TAR_EPOCH)
        self.tar_data = self.tar_data.reshape(self.tar_data.shape[0], self.tar_data.shape[1])
        self.max_depth = max_depth
        self.min_cluster_size = min_cluster_size




    def gen_center(self,data,k=10):
        """
        use keans 
        """
        kmeans = KMeans(n_clusters=k)  
        kmeans.fit(data)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        radii = []
        for i in range(k):
            cluster_data = data[labels == i]
            if len(cluster_data) > 0:
                # calculate each sample distance to center
                distances = np.sqrt(((cluster_data - centers[i]) ** 2).sum(axis=1))
                radii.append(np.max(distances))
            else:
                radii.append(0)
        
        return centers,labels,radii
    
    def compute_center(self, data):
        """Compute the center (mean) of the data."""
        return np.mean(data, axis=0)

    def initialize_proxies(self):
        ref_data = self.ref_data_provider.train_representation(self.REF_EPOCH)
        tar_data = self.tar_data_provider.train_representation(self.TAR_EPOCH)
        
        ref_center = self.compute_center(ref_data)
        tar_center = self.compute_center(tar_data)

        # Initialize proxies with the computed centers
        ref_proxy = [ref_center]
        tar_proxy = [tar_center]

        return ref_proxy, tar_proxy
  

    def similarity(self, pred1, pred2):
        """Compute similarity based on variance across the normalized softmax predictions."""
        scaler = MinMaxScaler()
        pred1_normalized = scaler.fit_transform(pred1.reshape(-1, 1)).ravel()
        pred2_normalized = scaler.fit_transform(pred2.reshape(-1, 1)).ravel()
    
        return -np.sum((pred1_normalized - pred2_normalized) ** 2)

    def align_centers_by_similarity(self, ref_centers, tar_centers):
        """Align target centers to reference centers based on the similarity of their predictions."""
        aligned_tar_centers = []
        aligned_tar_labels = []
        ref_preds = self.ref_data_provider.get_pred(self.REF_EPOCH, ref_centers)
        tar_preds = self.tar_data_provider.get_pred(self.TAR_EPOCH, tar_centers)
    
        for ref_pred in ref_preds:
            similarities = [self.similarity(ref_pred, tar_pred) for tar_pred in tar_preds]
            best_match_index = np.argmax(similarities)
            
            aligned_tar_centers.append(tar_centers[best_match_index])
            aligned_tar_labels.append(best_match_index)
        
        return aligned_tar_centers, aligned_tar_labels



    def align_clusters(self, ref_centers, tar_centers):
        """
        For the input calculate the similarity and align the centers
        """
        ref_preds = self.ref_data_provider.get_pred(self.REF_EPOCH, ref_centers)
        tar_preds = self.tar_data_provider.get_pred(self.TAR_EPOCH, tar_centers)
        
        ref_order = []
        for r_pred in ref_preds:
            similarities = [self.similarity(r_pred, t_pred) for t_pred in tar_preds]
            ref_order.append(np.argmax(similarities))
        
        return np.array(tar_centers)[ref_order]

    def recursive_align_and_cluster(self, ref_data, tar_data, depth=0,max_depth=10):
        """
        Recursively align and cluster the data based on their predictions' similarity.
    
        :param ref_data: Reference data to be clustered.
        :param tar_data: Target data to be aligned and clustered with the reference data.
        :param depth: Current depth of recursion.
        :param max_depth: Maximum allowed depth for recursion.
        :return: Two lists of proxies for reference and target data.
        """

         # Initialize lists to store the proxies for the current depth
        ref_proxies = []
        tar_proxies = []
        print("depth:", depth)
        if depth >= self.max_depth or len(ref_data) < self.min_cluster_size or len(tar_data) < self.min_cluster_size:
            return [], []

        # Obtain centers and labels from clustering of the reference and target data
        ref_centers, ref_labels, _ = self.gen_center(ref_data)
        tar_centers, tar_labels, _ = self.gen_center(tar_data)

        # Align target centers based on the similarity with reference centers
        aligned_tar_centers, aligned_tar_label_indices = self.align_centers_by_similarity(ref_centers, tar_centers)

        # Iterate over each aligned pair of centers (reference and target)
        for ref_center, ref_label, aligned_tar_label_index in zip(ref_centers, ref_labels, aligned_tar_label_indices):
        
            # Extract data corresponding to the current cluster label
            ref_cluster_data = ref_data[ref_labels == ref_label]
            tar_cluster_data = tar_data[tar_labels == aligned_tar_label_index]
        
            # Recursively align and cluster the data inside the current cluster
            ref_sub_proxies, tar_sub_proxies = self.recursive_align_and_cluster(ref_cluster_data, tar_cluster_data, depth + 1, self.max_depth)
        
            # If there are sub-proxies, extend the proxy list; otherwise, add the current center
            ref_proxies.extend(ref_sub_proxies if ref_sub_proxies else [ref_center])
            tar_proxies.extend(tar_sub_proxies if tar_sub_proxies else [aligned_tar_centers[aligned_tar_label_index]])
            # print("shape",len(ref_proxies),len(tar_proxies))

        # ref_preds = self.ref_data_provider.get_pred(self.REF_EPOCH, np.array(ref_proxies)).argmax(axis=1)
        # tar_preds = self.tar_data_provider.get_pred(self.TAR_EPOCH, np.array(tar_proxies)).argmax(axis=1)
        # print("pred res",ref_preds,ref_preds,tar_preds,tar_preds)

        return ref_proxies, tar_proxies

    def generate_proxies(self):

        return self.recursive_align_and_cluster(self.ref_data, self.tar_data,0,self.max_depth)

    


class AlignedSkeletonGenerator_DBSCAN(AlignedSkeletonGeneratorAbstractClass):
    def __init__(self, ref_data_provider, tar_data_provider, REF_EPOCH, TAR_EPOCH,max_depth=10,min_cluster_size=800) -> None:
        self.ref_data_provider = ref_data_provider
        self.tar_data_provider = tar_data_provider
        self.REF_EPOCH = REF_EPOCH
        self.TAR_EPOCH = TAR_EPOCH
        self.ref_data = self.ref_data_provider.train_representation(self.REF_EPOCH)
        self.ref_data = self.ref_data.reshape(self.ref_data.shape[0], self.ref_data.shape[1])
        self.tar_data = self.tar_data_provider.train_representation(self.TAR_EPOCH)
        self.tar_data = self.tar_data.reshape(self.tar_data.shape[0], self.tar_data.shape[1])
        self.max_depth = max_depth
        self.min_cluster_size = min_cluster_size

    def gen_center(self, data, eps=0.5, min_samples=5):
        """
        Use DBSCAN for clustering and calculate the centers.
        """
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(data)

        unique_labels = set(labels)
        centers = []
        radii = []
        
        for label in unique_labels:
            if label == -1:
                # Skip noise points
                continue

            cluster_data = data[labels == label]
            if len(cluster_data) > 0:
                center = cluster_data.mean(axis=0)
                centers.append(center)
                # Calculate each sample distance to center
                distances = np.sqrt(((cluster_data - center) ** 2).sum(axis=1))
                radii.append(np.max(distances))
            else:
                radii.append(0)
                
        return centers, labels, radii
    
    def compute_center(self, data):
        """Compute the center (mean) of the data."""
        return np.mean(data, axis=0)

    def initialize_proxies(self):
        ref_data = self.ref_data_provider.train_representation(self.REF_EPOCH)
        tar_data = self.tar_data_provider.train_representation(self.TAR_EPOCH)
        
        ref_center = self.compute_center(ref_data)
        tar_center = self.compute_center(tar_data)

        # Initialize proxies with the computed centers
        ref_proxy = [ref_center]
        tar_proxy = [tar_center]

        return ref_proxy, tar_proxy
  

    def similarity(self, pred1, pred2):
        """Compute similarity based on variance across the normalized softmax predictions."""
        scaler = MinMaxScaler()
        pred1_normalized = scaler.fit_transform(pred1.reshape(-1, 1)).ravel()
        pred2_normalized = scaler.fit_transform(pred2.reshape(-1, 1)).ravel()
    
        return -np.sum((pred1_normalized - pred2_normalized) ** 2)

    def align_centers_by_similarity(self, ref_centers, tar_centers):
        """Align target centers to reference centers based on the similarity of their predictions."""
        aligned_tar_centers = []
        aligned_tar_labels = []
        ref_preds = self.ref_data_provider.get_pred(self.REF_EPOCH, ref_centers)
        tar_preds = self.tar_data_provider.get_pred(self.TAR_EPOCH, tar_centers)
    
        for ref_pred in ref_preds:
            similarities = [self.similarity(ref_pred, tar_pred) for tar_pred in tar_preds]
            best_match_index = np.argmax(similarities)
            
            aligned_tar_centers.append(tar_centers[best_match_index])
            aligned_tar_labels.append(best_match_index)
        
        return aligned_tar_centers, aligned_tar_labels



    def align_clusters(self, ref_centers, tar_centers):
        """
        For the input calculate the similarity and align the centers
        """
        ref_preds = self.ref_data_provider.get_pred(self.REF_EPOCH, ref_centers)
        tar_preds = self.tar_data_provider.get_pred(self.TAR_EPOCH, tar_centers)
        
        ref_order = []
        for r_pred in ref_preds:
            similarities = [self.similarity(r_pred, t_pred) for t_pred in tar_preds]
            ref_order.append(np.argmax(similarities))
        
        return np.array(tar_centers)[ref_order]

    def recursive_align_and_cluster(self, ref_data, tar_data, depth=0,max_depth=10):
        """
        Recursively align and cluster the data based on their predictions' similarity.
    
        :param ref_data: Reference data to be clustered.
        :param tar_data: Target data to be aligned and clustered with the reference data.
        :param depth: Current depth of recursion.
        :param max_depth: Maximum allowed depth for recursion.
        :return: Two lists of proxies for reference and target data.
        """

         # Initialize lists to store the proxies for the current depth
        ref_proxies = []
        tar_proxies = []
        print("depth:", depth)
        if depth >= self.max_depth or len(ref_data) < self.min_cluster_size or len(tar_data) < self.min_cluster_size:
            return [], []

        # Obtain centers and labels from clustering of the reference and target data
        ref_centers, ref_labels, _ = self.gen_center(ref_data)
        tar_centers, tar_labels, _ = self.gen_center(tar_data)

        # Align target centers based on the similarity with reference centers
        aligned_tar_centers, aligned_tar_label_indices = self.align_centers_by_similarity(ref_centers, tar_centers)

        # Iterate over each aligned pair of centers (reference and target)
        for ref_center, ref_label, aligned_tar_label_index in zip(ref_centers, ref_labels, aligned_tar_label_indices):
        
            # Extract data corresponding to the current cluster label
            ref_cluster_data = ref_data[ref_labels == ref_label]
            tar_cluster_data = tar_data[tar_labels == aligned_tar_label_index]
        
            # Recursively align and cluster the data inside the current cluster
            ref_sub_proxies, tar_sub_proxies = self.recursive_align_and_cluster(ref_cluster_data, tar_cluster_data, depth + 1, self.max_depth)
        
            # If there are sub-proxies, extend the proxy list; otherwise, add the current center
            ref_proxies.extend(ref_sub_proxies if ref_sub_proxies else [ref_center])
            tar_proxies.extend(tar_sub_proxies if tar_sub_proxies else [aligned_tar_centers[aligned_tar_label_index]])
            # print("shape",len(ref_proxies),len(tar_proxies))

        # ref_preds = self.ref_data_provider.get_pred(self.REF_EPOCH, np.array(ref_proxies)).argmax(axis=1)
        # tar_preds = self.tar_data_provider.get_pred(self.TAR_EPOCH, np.array(tar_proxies)).argmax(axis=1)
        # print("pred res",ref_preds,ref_preds,tar_preds,tar_preds)

        return ref_proxies, tar_proxies

    def generate_proxies(self):

        return self.recursive_align_and_cluster(self.ref_data, self.tar_data,0,self.max_depth)

    

    

        
