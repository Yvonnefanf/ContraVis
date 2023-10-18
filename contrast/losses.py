import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from contrast.CKA_utils import *
import numpy as np
import torch.nn.functional as F
from singleVis.utils import *
import os
import sys
sys.path.append('..')



class KNNOverlapLoss(nn.Module):
    def __init__(self, k=5):
        super(KNNOverlapLoss, self).__init__()
        self.k = k
    
    def forward(self, input, target):
        input_np = input.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        # Compute K-nearest neighbors for input and target
        knn_input = NearestNeighbors(n_neighbors=self.k).fit(input_np)
        knn_target = NearestNeighbors(n_neighbors=self.k).fit(target_np)

        input_neighbors = knn_input.kneighbors(input_np, return_distance=False)
        target_neighbors = knn_target.kneighbors(target_np, return_distance=False)

        # Calculate the overlap between input_neighbors and target_neighbors
        overlap = 0
        for i in range(input_np.shape[0]):
            common_neighbors = np.intersect1d(input_neighbors[i], target_neighbors[i]).shape[0]
            overlap += common_neighbors

        # Calculate the KNN overlap loss
        loss = 1 - (overlap / (input_np.shape[0] * self.k))

        return torch.tensor(loss, device=input.device, dtype=torch.float32, requires_grad=True)


class CKALoss(nn.Module):
    def __init__(self, gamma=None, alpha=1e-3):
        super(CKALoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, X, Y, Z):
        K_xx = kernel_HSIC(X, X, self.gamma)
        K_yy = kernel_HSIC(Y, Y, self.gamma)
        K_zz = kernel_HSIC(Z, Z, self.gamma)
        K_xy = kernel_HSIC(X, Y, self.gamma)
        K_xz = kernel_HSIC(X, Z, self.gamma)
        K_yz = kernel_HSIC(Y, Z, self.gamma)
        
        cka_loss1 = 1 - torch.mean(K_xy) / torch.sqrt(torch.mean(K_xx) * torch.mean(K_yy))
        cka_loss2 = 1 - torch.mean(K_yz) / torch.sqrt(torch.mean(K_yy) * torch.mean(K_zz))
        
        loss = cka_loss1 + self.alpha * cka_loss2
        return loss
    

class PredictionLoss(nn.Module):
    def __init__(self, tar_model, ref_model, tar_provider, ref_provider, TAR_EPOCH, REF_EPOCH, DEVICE, alpha_for_pred_ref):
        super(PredictionLoss, self).__init__()
        self.tar_model = tar_model
        self.ref_model = ref_model
        self.tar_provider = tar_provider
        self.ref_provider = ref_provider
        self.alpha_for_pred_ref = alpha_for_pred_ref
        self.TAR_EPOCH =TAR_EPOCH
        self.REF_EPOCH = REF_EPOCH
        self.DEVICE = DEVICE
        self.split = -1

    def prediction_function(self, Epoch, model_path, model):
        #TODO
        model_location = os.path.join(model_path, "Model", "Epoch_{:d}".format(Epoch), "subject_model.pth")
        model.load_state_dict(torch.load(model_location, map_location=torch.device("cpu")))
        model.to(self.DEVICE)
        model.eval()

        model = torch.nn.Sequential(*(list(model.children())[self.split:]))
        model.to(self.DEVICE)
        model.eval()
      
        return model

    def get_pred(self, epoch, data, model_path, model):
        '''
        get the prediction score for data in epoch_id
        :param data: numpy.ndarray
        :param epoch_id:
        :return: pred, numpy.ndarray
        '''
        prediction_func = self.prediction_function(epoch,model_path, model)

        # data = torch.from_numpy(data)
        data = data.to(self.DEVICE)
        pred = batch_run(prediction_func, data)
        return pred.squeeze()


    def forward(self, adjusted_input, init_input):
        target_output = self.tar_provider.get_pred(self.TAR_EPOCH, init_input)
        # tar_output = self.get_pred(self.TAR_EPOCH, adjusted_input, self.tar_provider.content_path, self.tar_model)
        ref_output = self.get_pred(self.REF_EPOCH, adjusted_input, self.ref_provider.content_path, self.ref_model)
        
        # loss_tar_output = F.mse_loss(torch.tensor(tar_output), torch.tensor(target_output))
        loss_ref_output = F.mse_loss(torch.tensor(ref_output), torch.tensor(target_output))
        loss_Rep = F.mse_loss(adjusted_input, torch.tensor(init_input))
        
        # loss = loss_tar_output + loss_Rep + self.alpha_for_pred_ref * loss_ref_output
        loss =  loss_Rep + self.alpha_for_pred_ref * loss_ref_output
        return loss

class ConfidenceLoss(nn.Module):
    def __init__(self):
        super(ConfidenceLoss, self).__init__()

    def forward(self, tar_probs, ref_probs):
        # Get the confidence (maximum probability) for each prediction
        tar_confidence = torch.max(tar_probs, dim=1)[0]
        ref_confidence = torch.max(ref_probs, dim=1)[0]

        # Calculate the absolute difference between confidences
        confidence_diff = torch.abs(tar_confidence - ref_confidence)

        # Calculate the mean absolute difference as the loss
        loss = torch.mean(confidence_diff)
        return loss

class TopoLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(TopoLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, y_trans, y):
        # Compute pairwise distances between original and mapped outputs
        dist_y = pairwise_distances(y)
        dist_y_trans = pairwise_distances(y_trans)
        
        # Compute pairwise differences between distances
        delta_dist = dist_y - dist_y_trans
        
        # Compute pairwise similarities between neighborhoods
        neigh_y = compute_neighborhoods(dist_y)
        neigh_y_trans = compute_neighborhoods(dist_y_trans)
        sim = compute_neighborhood_similarity(neigh_y, neigh_y_trans)
        
        # Compute topology-preserving loss
        topo_loss = self.alpha * torch.sum(torch.square(delta_dist)) + self.beta * torch.sum(torch.square(1.0 - sim))
        
        return topo_loss
        
def pairwise_distances(x):
    # Compute pairwise distances between rows of matrix x
    norm = (x ** 2).sum(1).reshape(-1, 1)
    dist = norm + norm.T - 2.0 * torch.mm(x, x.T)
    return torch.sqrt(torch.clamp(dist, 0.0, np.inf))

def compute_neighborhoods(dist):
    # Compute neighborhoods based on pairwise distances
    k = 5 # number of nearest neighbors
    idx = torch.argsort(dist, dim=1)
    neigh = idx[:, 1:k+1]
    return neigh

def compute_neighborhood_similarity(neigh1, neigh2):
    # Compute similarity between neighborhoods
    sim = torch.mean((neigh1.unsqueeze(2) == neigh2.unsqueeze(1)).float(), dim=2)
    return sim




