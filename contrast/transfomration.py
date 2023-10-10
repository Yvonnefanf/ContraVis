"""
Mapper: transfomrmation different representations
high d to high d
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import NearestNeighbors
import numpy as np

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
    def __init__(self,ref_data,tar_data,device) -> None:
        self.ref_data = ref_data
        self.tar_data = tar_data
        self.device = device
        self.model = Autoencoder().to(device)
       

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
        return array[indices]

    

    
    def transformation_train(self,lambda_translation=0.5,num_epochs = 100,lr=0.001):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
         # Assume tar_tensor and ref_tensor are your data in PyTorch tensor format
        tar_tensor = torch.tensor(self.tar_data, dtype=torch.float32).cuda() 
        ref_tensor = torch.tensor(self.ref_data, dtype=torch.float32).cuda()
        tar_tensor = tar_tensor.to(self.device)
        ref_tensor = ref_tensor.to(self.device)
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
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

        # To get the mapped version of tar
        tar_mapped = self.model.encoder(tar_tensor).cpu().detach().numpy()

        # To get ref back from the mapped tar
        ref_reconstructed = self.model.decoder(torch.tensor(tar_mapped).to(self.device)).cpu().detach().numpy()

        return self.model, tar_mapped, ref_reconstructed
    

    def compute_neighbors(self, data, n_neighbors=15):
        nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='auto').fit(data) # 加1是因为第一个邻居是数据点自己
        _, indices = nbrs.kneighbors(data)
        return indices[:, 1:] # 去掉第一个邻居，因为它是数据点自己

    def similarity_loss(self, mapped_data, original_neighbors):
        mapped_neighbors = self.compute_neighbors(mapped_data.cpu().detach().numpy())

        # 计算匹配度，这里使用集合的交集大小
        match_score = 0
        for m_neigh, o_neigh in zip(mapped_neighbors, original_neighbors):
            match_score += len(set(m_neigh) & set(o_neigh))
        
        # 最大匹配度是15，因此损失是15减去实际匹配度
        loss = (15 * len(mapped_data) - match_score) / len(mapped_data)
        return torch.tensor(loss, device=self.device)
    
    def transformation_train_advanced(self,lambda_translation=0.5, lambda_similarity=1.0,num_epochs=100,lr=0.001,sample_size=1000):
        original_neighbors = self.compute_neighbors(self.tar_data)
    
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
         # Assume tar_tensor and ref_tensor are your data in PyTorch tensor format
        tar_tensor = torch.tensor(self.tar_data, dtype=torch.float32).cuda() 
        ref_tensor = torch.tensor(self.ref_data, dtype=torch.float32).cuda()
        tar_tensor = tar_tensor.to(self.device)
        ref_tensor = ref_tensor.to(self.device)

         # Train the autoencoder
        for epoch in range(200):
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
    


    

