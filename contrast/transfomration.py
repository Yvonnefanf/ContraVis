"""
Mapper: transfomrmation different representations

"""

import torch
import torch.nn as nn
import torch.optim as optim

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
    


def transformation_train(ref_data,tar_data):
    # Initialize the model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Assume tar_tensor and ref_tensor are your data in PyTorch tensor format
    tar_tensor = torch.tensor(tar_data, dtype=torch.float32).cuda()  # X_train2 was the tar in your previous example
    ref_tensor = torch.tensor(ref_data, dtype=torch.float32).cuda()  # X_train1 was the ref in your previous example
    tar_tensor = tar_tensor.to(device)
    ref_tensor = ref_tensor.to(device)
    lambda_translation = 0.5

    # Train the autoencoder
    num_epochs = 200
    for epoch in range(num_epochs):
        model.train()

        optimizer.zero_grad()

        # Forward pass
        latent_representation = model.encoder(tar_tensor)
        outputs = model.decoder(latent_representation)

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
    tar_mapped = model.encoder(tar_tensor).cpu().detach().numpy()

    # To get ref back from the mapped tar
    ref_reconstructed = model.decoder(torch.tensor(tar_mapped).to(device)).cpu().detach().numpy()

    return model, tar_mapped, ref_reconstructed



