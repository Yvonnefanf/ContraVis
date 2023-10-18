import torch.optim as optim
import numpy as np
from pyemd import emd
import torch.nn as nn
from scipy.spatial.distance import cdist
from sklearn.neighbors import kneighbors_graph
import torch.nn.functional as F
import sys
sys.path.append('..')
import torch
from contrast.losses import KNNOverlapLoss, CKALoss, PredictionLoss, ConfidenceLoss




def kl_div_loss(P, Q):
    # Check if the input contains NaN, infinity, or very large values
    if not torch.isfinite(P).all() or not torch.isfinite(Q).all() or not torch.max(torch.abs(P)).item() < np.finfo(np.float32).max or not torch.max(torch.abs(Q)).item() < np.finfo(np.float32).max:
        # Replace NaN, infinity, or very large values with zeros
        P[~torch.isfinite(P) | (torch.abs(P) >= np.finfo(np.float32).max)] = 0
        Q[~torch.isfinite(Q) | (torch.abs(Q) >= np.finfo(np.float32).max)] = 0

    kl_divergence = torch.sum(P * torch.log(P / Q))

    # Calculate KL divergence between Q and P
    kl_divergence += torch.sum(Q * torch.log(Q / P))

    # Return the average KL divergence
    return kl_divergence.mean()

def gaussian_kernel(x1, x2, sigma=1.0):
    x1 = x1.unsqueeze(1)  # shape: (n1, 1, d)
    x2 = x2.unsqueeze(0)  # shape: (1, n2, d)
    dist = torch.sum((x1 - x2) ** 2, dim=-1)  # shape: (n1, n2)
    return torch.exp(-dist / (2 * sigma ** 2))

def mmd_loss(x1, x2, kernel=gaussian_kernel, sigma=1.0):
    n1, n2 = x1.size(0), x2.size(0)

    k11 = kernel(x1, x1, sigma=sigma).sum() / (n1 * (n1 - 1))
    k22 = kernel(x2, x2, sigma=sigma).sum() / (n2 * (n2 - 1))
    k12 = kernel(x1, x2, sigma=sigma).sum() / (n1 * n2)

    return k11 + k22 - 2 * k12



def earth_movers_distance(X, Y, k=5):
    X, Y = X.detach().numpy(), Y.detach().numpy()
    
    # Compute KNN graphs
    X_knn_graph = kneighbors_graph(X, k, mode='distance')
    Y_knn_graph = kneighbors_graph(Y, k, mode='distance')
    
    # Convert to dense NumPy arrays
    X_knn_matrix = X_knn_graph.toarray()
    Y_knn_matrix = Y_knn_graph.toarray()

    # Calculate the EMD between the KNN distance matrices
    distance_matrix = cdist(X_knn_matrix, Y_knn_matrix)
    first_histogram = np.ones(X_knn_matrix.shape[0]) / X_knn_matrix.shape[0]
    second_histogram = np.ones(Y_knn_matrix.shape[0]) / Y_knn_matrix.shape[0]

    loss = emd(first_histogram, second_histogram, distance_matrix)
    loss_tensor = torch.tensor(loss, requires_grad=True)

    return loss_tensor

def frobenius_norm_loss(predicted, target):
    return torch.norm(predicted - target, p='fro') / predicted.numel()

def prediction_loss(trans_X, Y):
    
    target_output = tar_provider.get_pred(TAR_EPOCH, Y.detach().numpy())
    # tar_output = self.get_pred(self.TAR_EPOCH, adjusted_input, self.tar_provider.content_path, self.tar_model)
    ref_output = tar_provider.get_pred(TAR_EPOCH, trans_X.detach().numpy())

    loss_ref_output = F.mse_loss(torch.tensor(ref_output), torch.tensor(target_output))
    loss_Rep = F.mse_loss(trans_X, Y)
        
    # loss = loss_tar_output + loss_Rep + self.alpha_for_pred_ref * loss_ref_output
    loss =  loss_Rep + 1 * loss_ref_output
    return loss

# Define hyperparameters
num_epochs = 10
batch_size = 100
learning_rate = 1e-4

# Define the loss function and the optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate,weight_decay=1e-5)

alpha = 1 # weight for topological loss, adjust this according to your requirements


autoencoder = SimpleAutoencoder(input_dim,output_dim)
########### load pre autoencoder
checkpoint = torch.load("/home/yifan/projects/deepdebugertool/DLVisDebugger/AlignVisAutoEncoder/checkpoints/cak_1v1.pth")
autoencoder.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

######### train sample + generated boundary sample's => input  #############
from AlignVisAutoEncoder.data_loader import DataLoaderInit

indicates = np.random.choice(np.arange(50000), size=5000, replace=False)

input_X = np.concatenate((ref_provider.train_representation(REF_EPOCH)[indicates], ref_features[indices]),axis=0)
input_Y = np.concatenate((tar_provider.train_representation(TAR_EPOCH)[indicates], tar_features[indices]),axis=0)
data_loader_b = DataLoaderInit(input_X, input_Y, batch_size)
dataloader_b = data_loader_b.get_data_loader()

# Training loop
for epoch in range(num_epochs):
    # Initialize a list to store the predictions of unlabelled data
    unlabelled_preds = []
    for data_X, data_Y in dataloader_b: # Assuming you have a DataLoader instance with paired data (X, Y)
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass (encoding Y and decoding to X's space)
        transformed_Y = autoencoder.encoder(data_Y)
        recon_X = autoencoder.decoder(transformed_Y)
        transformed_X = autoencoder.decoder(data_X)

        ##### knn loss
        knn_overlap_loss = KNNOverlapLoss(k=15)

        knn_loss_encoder = knn_overlap_loss(input=transformed_Y, target=data_Y)

        knn_loss_decoder = knn_overlap_loss(input=recon_X, target=data_Y)

        # knn_loss = knn_overlap_loss(input=transformed_X, target=data_Y)
        ###### make ref' s distribution like tar
        mmd_loss_v = mmd_loss(recon_X,data_Y)
        # dis_loss = 0
        
        loss_f_decoder = knn_loss_decoder
        loss_f_encoder = knn_loss_encoder

        pred_loss = prediction_loss(recon_X, data_Y)

        

        #### CKA loss
        cka_loss_f = CKALoss(gamma=None, alpha=1e-8)
        cka_loss = cka_loss_f(data_Y,transformed_Y,recon_X)

        loss = loss_f_decoder + loss_f_encoder + pred_loss + cka_loss + 20 * mmd_loss_v

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()


    # Print the loss for each epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Loss decoder: {loss_f_decoder.item():.4f},Loss encoder: {loss_f_encoder.item():.4f},pred_loss,{pred_loss.item():.4f},CKA,{cka_loss.item():.4f},mmd_loss:{mmd_loss_v}')

torch.save({
    'epoch': TAR_EPOCH,
    'model_state_dict': autoencoder.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss
}, "/home/yifan/projects/deepdebugertool/DLVisDebugger/AlignVisAutoEncoder/checkpoints/drop_only_boundary.pth")