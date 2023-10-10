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
       

    def random_sample(self, tensor, sample_size):
        """
        Randomly sample a subset from a tensor.

        Args:
        - tensor (torch.Tensor): The input tensor to sample from.
        - sample_size (int): The size of the random sample.

        Returns:
        - torch.Tensor: A subset of the input tensor.
        """
        indices = torch.randperm(tensor.size(0))[:sample_size]
        return tensor[indices]

    
    def compute_similarity_matrix(self, data, sample_size=None):
        if sample_size:
            data = self.random_sample(data, sample_size)
        
        nbrs = NearestNeighbors(n_neighbors=16, algorithm='auto').fit(data)
        _, indices = nbrs.kneighbors(data)
        return torch.tensor(indices, device=self.device)
    
    def similarity_loss(self, mapped_data, sample_size=None):
        # Here, we also consider a subset for the mapped_data when computing the loss
        if sample_size:
            mapped_data = self.random_sample(mapped_data, sample_size)
        
        mapped_nearest = torch.argsort(torch.cdist(mapped_data, mapped_data, p=2), dim=1)[:, 1:16]
        original_nearest = self.compute_similarity_matrix(mapped_data.cpu().detach().numpy(), sample_size)
        
        loss = nn.MSELoss()(mapped_nearest.float(), original_nearest.float())
        return loss
    
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
    
    def transformation_train_advanced(self,lambda_translation=0.5, lambda_similarity=1.0,num_epochs = 200,lr=0.001,sample_size=1000):
        self.similarity_matrix = self.compute_similarity_matrix(self.tar_data)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
         # Assume tar_tensor and ref_tensor are your data in PyTorch tensor format
        tar_tensor = torch.tensor(self.tar_data, dtype=torch.float32).cuda() 
        ref_tensor = torch.tensor(self.ref_data, dtype=torch.float32).cuda()
        tar_tensor = tar_tensor.to(self.device)
        ref_tensor = ref_tensor.to(self.device)
        # Train the autoencoder
        num_epochs = 200
        for epoch in range(num_epochs):
            self.model.train()

            optimizer.zero_grad()

            # Inside your training loop:
            latent_representation = self.model.encoder(tar_tensor)
            outputs = self.model.decoder(latent_representation)

            reconstruction_loss = criterion(outputs, tar_tensor)
            translation_loss = criterion(latent_representation, ref_tensor)
            # neighbor_loss = self.similarity_loss(latent_representation)
            neighbor_loss = self.similarity_loss(latent_representation, sample_size)

            # Combine the losses
            loss = reconstruction_loss + lambda_translation * translation_loss + lambda_similarity * neighbor_loss
            
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
    


    

