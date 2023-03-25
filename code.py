#importing the essential librararies and functions

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor, Normalize, Compose
import matplotlib.pyplot as plt


# Define the autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=7)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Define the transform
transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])

# Load the FashionMNIST dataset
fdataset = FashionMNIST(root="./data", train=True, download=True, transform=transform)

# Create a DataLoader
dataloader = DataLoader(fdataset, batch_size=64, shuffle=True, num_workers=4)

# Initialize the autoencoder
autoencoder = Autoencoder()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

# Train the autoencoder on 20 epochs
num_epochs = 20
for epoch in range(num_epochs):
    for batch_idx, (data, _) in enumerate(dataloader):
        optimizer.zero_grad()
        recon_data = autoencoder(data)
        loss = criterion(recon_data, data)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            axes[0].imshow(data[0].permute(1, 2, 0).numpy())
            axes[0].set_title("Input")
            axes[1].imshow(recon_data[0].detach().permute(1, 2, 0).numpy())
            axes[1].set_title("Output")
            #visualising the predictions
            
            plt.show()
