"""
Mapper: transfomrmation different representations
high d to high d
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import NearestNeighbors
import numpy as np
import sys
sys.path.append('..')
import torch
from contrast.losses import KNNOverlapLoss, CKALoss, PredictionLoss, ConfidenceLoss



# Define the autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim=512, encoding_dim=128):
        super(Autoencoder, self).__init__()
        
        # Encoder: tar to ref
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, input_dim),
            nn.ReLU()
        )
        
        # Decoder: ref to tar
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, input_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class TransformationTrainer():
    def __init__(self,ref_data,tar_data,ref_proxy, tar_proxy, ref_provider,tar_provider,ref_epoch,tar_epoch,device) -> None:
        self.ref_data = ref_data
        self.tar_data = tar_data
        self.ref_proxy = ref_proxy
        self.tar_proxy = tar_proxy
        self.ref_provider = ref_provider
        self.tar_provider = tar_provider
        self.ref_epoch = ref_epoch
        self.tar_epoch= tar_epoch
        self.device = device
        self.model = Autoencoder().to(device)
        self.ref_data = self.ref_data.reshape(self.ref_data.shape[0],self.ref_data.shape[1])
        self.tar_data = self.tar_data.reshape(self.tar_data.shape[0],self.tar_data.shape[1])
        self.tar_pred = self.tar_provider.get_pred(self.tar_epoch, self.tar_data)
        self.ref_pred = self.ref_provider.get_pred(self.ref_epoch, self.ref_data)
    
    def random_sample(self, array, sample_size):
        """
        Randomly sample a subset from a numpy array.

        Args:
        - array (numpy.ndarray): The input array to sample from.
        - sample_size (int): The size of the random sample.

        Returns:
        - numpy.ndarray: A subset of the input array.
        """
        indices = np.random.permutation(array.shape[0])[:sample_size]
        return array[indices], indices
    
    def filter_diff(self):
        diff_indicates = []

        tar_pred = self.tar_pred.argmax(axis=1)
        ref_pred = self.ref_pred.argmax(axis=1)
        for i in range(len(tar_pred)):
            if tar_pred[i] != ref_pred[i]:
                diff_indicates.append(i)
        return diff_indicates
    
    def euclidean_distance(self, vec1, vec2):
        return np.linalg.norm(vec1 - vec2)
    
    def align_ref_data(self,sample_size=500):
        
        diff_indicates = self.filter_diff()
        print("diff is", len(diff_indicates))
        tar_pred = self.tar_pred
        ref_pred = self.ref_pred
        for i in diff_indicates:
            # Randomly sample from ref_pred
            sampled_ref_pred, sampled_indices = self.random_sample(ref_pred, sample_size)
            # calculate the distance
            # calculate the distance only with the sampled ref points
            distances = [self.euclidean_distance(tar_pred[i], sampled_ref_pred[j]) for j in range(sample_size)]
            # find the most similar point
            similar_ref_index = np.argmin(distances)
            # replace
            self.ref_data[i] = self.ref_data[similar_ref_index]
    

    
    def transformation_train(self,lambda_translation=0.5,num_epochs = 100,lr=0.001):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # Assume tar_tensor and ref_tensor are your data in PyTorch tensor format
        self.align_ref_data()
        print("finished aligned")
        self.tar_train = np.concatenate((self.tar_data, self.tar_proxy),axis=0)
        self.ref_train = np.concatenate((self.ref_data, self.ref_proxy),axis=0)
        tar_tensor = torch.tensor(self.tar_train, dtype=torch.float32).cuda() 
        ref_tensor = torch.tensor(self.ref_train, dtype=torch.float32).cuda()
        tar_tensor = tar_tensor.to(self.device)
        ref_tensor = ref_tensor.to(self.device)
        # knn_overlap_loss = KNNOverlapLoss(k=15)

        # Train the autoencoder
        for epoch in range(num_epochs):
            self.model.train()

            optimizer.zero_grad()

            # Forward pass
            latent_representation = self.model.encoder(tar_tensor)
            outputs = self.model.decoder(latent_representation)

            # Compute the two losses
            reconstruction_loss = criterion(outputs, tar_tensor)
            translation_loss = criterion(latent_representation, ref_tensor)

            # Combine the losses
            loss = reconstruction_loss + lambda_translation * translation_loss
            # + lambda_knn * (knn_loss_encoder + knn_loss_decoder) 
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

        # To get the mapped version of tar
        tar_mapped = self.model.encoder(tar_tensor).cpu().detach().numpy()

        tar_data_shape = self.tar_data.shape[0]
        tar_data_mapped = tar_mapped[:tar_data_shape]
        tar_proxy_mapped = tar_mapped[tar_data_shape:]

        

        # To get ref back from the mapped tar
        ref_reconstructed = self.model.decoder(torch.tensor(tar_mapped).to(self.device)).cpu().detach().numpy()

        return self.model, tar_data_mapped, tar_proxy_mapped, ref_reconstructed
    

    def compute_neighbors(self, data, n_neighbors=15):
        nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='auto').fit(data) # 加1是因为第一个邻居是数据点自己
        _, indices = nbrs.kneighbors(data)
        return indices[:, 1:] # 去掉第一个邻居，因为它是数据点自己

    def similarity_loss(self, mapped_data, original_neighbors):
        mapped_neighbors = self.compute_neighbors(mapped_data.cpu().detach().numpy())

        # calculate the match score
        match_score = 0
        for m_neigh, o_neigh in zip(mapped_neighbors, original_neighbors):
            match_score += len(set(m_neigh) & set(o_neigh))
        
        # most match is 15 
        loss = (15 * len(mapped_data) - match_score) / len(mapped_data)
        return torch.tensor(loss, device=self.device)
    
    def transformation_train_advanced(self,lambda_translation=0.5, lambda_similarity=1.0,num_epochs=100,lr=0.001,base_epoch=50):
        original_neighbors = self.compute_neighbors(self.tar_data)
    
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
         # Assume tar_tensor and ref_tensor are your data in PyTorch tensor format
        tar_tensor = torch.tensor(self.tar_data, dtype=torch.float32).cuda() 
        ref_tensor = torch.tensor(self.ref_data, dtype=torch.float32).cuda()
        tar_tensor = tar_tensor.to(self.device)
        ref_tensor = ref_tensor.to(self.device)

         # Train the autoencoder
        for epoch in range(base_epoch):
            self.model.train()

            optimizer.zero_grad()

            # Forward pass
            latent_representation = self.model.encoder(tar_tensor)
            outputs = self.model.decoder(latent_representation)

            # Compute the two losses
            reconstruction_loss = criterion(outputs, tar_tensor)
            translation_loss = criterion(latent_representation, ref_tensor)

            # Combine the losses
            loss = reconstruction_loss + lambda_translation * translation_loss
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

        # Train the adv autoencoder
        for epoch in range(num_epochs):
            self.model.train()

            optimizer.zero_grad()

            # Inside your training loop:
            latent_representation = self.model.encoder(tar_tensor)
            outputs = self.model.decoder(latent_representation)

            reconstruction_loss = criterion(outputs, tar_tensor)
            translation_loss = criterion(latent_representation, ref_tensor)
            # neighbor_loss = self.similarity_loss(latent_representation)
            neighbor_loss = self.similarity_loss(latent_representation, original_neighbors)

            # Combine the losses
            loss = reconstruction_loss + lambda_translation * translation_loss + lambda_similarity * neighbor_loss
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # if epoch % 20 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}],reconstruction_loss:{reconstruction_loss.item():.4f},translation_loss:{translation_loss.item():.4f},neighbor_loss:{neighbor_loss.item():.4f}, Loss: {loss.item():.4f}")

        # To get the mapped version of tar
        tar_mapped = self.model.encoder(tar_tensor).cpu().detach().numpy()

        # To get ref back from the mapped tar
        ref_reconstructed = self.model.decoder(torch.tensor(tar_mapped).to(self.device)).cpu().detach().numpy()

        return self.model, tar_mapped, ref_reconstructed
    


    

