import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# Constants and global variables
LATENT_DIM = 256     # Latent space dimensionality
TIMESTEPS = 1000     # Number of diffusion steps
BETA_START = 0.0001  # Starting value of beta for noise scheduling
BETA_END = 0.02      # Ending value of beta for noise scheduling
LEARNING_RATE = 1e-4 # Learning rate for the model optimizer
EPOCHS = 100         # Number of epochs for training
BATCH_SIZE = 64      # Batch size for training
IMAGE_SIZE = 64      # Input image size (assumed square images)

# Compute betas, alphas, and alpha cumulative product for diffusion process
betas = torch.linspace(BETA_START, BETA_END, TIMESTEPS)
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# ----------------------------------
# Diffusion Model Architecture
# ----------------------------------

class DiffusionModel(nn.Module):
    """
    Diffusion Model that takes in latent representations
    and generates predictions for reverse diffusion.
    """
    def __init__(self, latent_dim):
        super(DiffusionModel, self).__init__()
        # Define model layers
        self.fc1 = nn.Linear(latent_dim, latent_dim * 2)
        self.fc2 = nn.Linear(latent_dim * 2, latent_dim * 2)
        self.fc3 = nn.Linear(latent_dim * 2, latent_dim)

    def forward(self, x, t):
        """
        Forward pass with timestep conditioning.
        :param x: Latent vector
        :param t: Timestep
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ----------------------------------
# Helper Functions for Diffusion
# ----------------------------------

def forward_diffusion_sample(x0, t, betas, alphas_cumprod):
    """
    Perform forward diffusion by adding noise to the data.
    :param x0: Original data
    :param t: Current timestep
    :param betas: Beta schedule
    :param alphas_cumprod: Cumulative product of alphas
    :return: Noisy data and added noise
    """
    noise = torch.randn_like(x0)
    noisy_data = alphas_cumprod[t] * x0 + (1 - alphas_cumprod[t]) * noise
    return noisy_data, noise


def reverse_diffusion(noisy_data, predicted_noise, t, betas):
    """
    Reverse diffusion process to remove noise from the noisy data.
    :param noisy_data: Noisy input data
    :param predicted_noise: Predicted noise by the model
    :param t: Timestep
    :param betas: Beta schedule
    :return: Denoised data
    """
    return noisy_data - betas[t] * predicted_noise


# ----------------------------------
# Training the Diffusion Model
# ----------------------------------

class ReverseDiffusion(nn.Module):
    """
    Reverse Diffusion class that utilizes a DiffusionModel
    to predict and remove noise from noisy data.
    """
    def __init__(self, model):
        super(ReverseDiffusion, self).__init__()
        self.model = model

    def reverse_diffusion(self, noisy_data, t):
        """
        Perform reverse diffusion using the model.
        :param noisy_data: Noisy data
        :param t: Timestep
        """
        predicted_noise = self.model(noisy_data, t)
        denoised_data = noisy_data - betas[t] * predicted_noise
        return denoised_data


# ----------------------------------
# Dataset and DataLoader (for images)
# ----------------------------------

class SimpleDataset(torch.utils.data.Dataset):
    """
    Simple dataset class for loading image data.
    Assumes images are stored in a 4D tensor (batch, channels, height, width).
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Create random dataset for simplicity (replace with real images)
dummy_data = torch.randn(1000, 3, IMAGE_SIZE, IMAGE_SIZE)
train_dataset = SimpleDataset(dummy_data)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# ----------------------------------
# Training Loop
# ----------------------------------

def train_model(model, train_loader, epochs, learning_rate):
    """
    Train the diffusion model.
    :param model: Diffusion model
    :param train_loader: DataLoader for training data
    :param epochs: Number of epochs
    :param learning_rate: Learning rate
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs = data.view(-1, LATENT_DIM)

            # Generate random timesteps and perform forward diffusion
            t = torch.randint(0, TIMESTEPS, (BATCH_SIZE,)).long()
            noisy_data, noise = forward_diffusion_sample(inputs, t, betas, alphas_cumprod)

            # Zero the gradients
            optimizer.zero_grad()

            # Predict noise
            predicted_noise = model(noisy_data, t)

            # Compute loss (difference between predicted and actual noise)
            loss = criterion(predicted_noise, noise)

            # Backpropagation and optimization step
            loss.backward()
            optimizer.step()

            # Track loss
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")


# Instantiate and train model
diffusion_model = DiffusionModel(latent_dim=LATENT_DIM)
train_model(diffusion_model, train_loader, EPOCHS, LEARNING_RATE)


# ----------------------------------
# Sampling Function
# ----------------------------------

def sample_from_model(model, latent_dim, num_samples=1):
    """
    Sample images from the trained diffusion model.
    :param model: Trained diffusion model
    :param latent_dim: Latent space dimensionality
    :param num_samples: Number of samples to generate
    :return: Generated images
    """
    with torch.no_grad():
        for i in range(num_samples):
            # Start from random noise
            x = torch.randn((1, latent_dim))

            # Perform reverse diffusion process
            for t in reversed(range(TIMESTEPS)):
                predicted_noise = model(x, t)
                x = reverse_diffusion(x, predicted_noise, t, betas)

            yield x


# ----------------------------------
# Sampling and Visualizing Results
# ----------------------------------

def visualize_samples(samples):
    """
    Visualize generated samples (assuming the samples are in image format).
    :param samples: Iterable of generated samples
    """
    import matplotlib.pyplot as plt

    for sample in samples:
        # Reshape latent vector into an image
        sample = sample.view(3, IMAGE_SIZE, IMAGE_SIZE).numpy()

        # Plot image
        plt.imshow(np.transpose(sample, (1, 2, 0)))
        plt.show()


# Generate samples from trained model
generated_samples = sample_from_model(diffusion_model, latent_dim=LATENT_DIM, num_samples=5)
visualize_samples(generated_samples)

