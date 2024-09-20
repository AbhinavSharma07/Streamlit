import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Latent Dimension of Diffusion Model
LATENT_DIM = 256  

# Number of Diffusion Steps (Timesteps)
TIMESTEPS = 1000  

# Beta Scheduler (Controls Noise Addition)
BETA_START = 0.0001
BETA_END = 0.02
betas = torch.linspace(BETA_START, BETA_END, TIMESTEPS)

# Precomputed variance values
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

class DiffusionModel(nn.Module):
    def __init__(self, latent_dim):
        super(DiffusionModel, self).__init__()
        # Define model layers
        self.fc1 = nn.Linear(latent_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, latent_dim)

    def forward(self, x, t):
        # Forward pass with timestep conditioning
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Forward Diffusion Process (Adding Noise)
def forward_diffusion_sample(x0, t):
    noise = torch.randn_like(x0)
    return alphas_cumprod[t] * x0 + (1 - alphas_cumprod[t]) * noise, noise

# Reverse Diffusion Process (Predict Noise and Restore Image)
class ReverseDiffusion(nn.Module):
    def __init__(self, model):
        super(ReverseDiffusion, self).__init__()
        self.model = model

    def reverse_diffusion(self, noisy_data, t):
        predicted_noise = self.model(noisy_data, t)
        return noisy_data - betas[t] * predicted_noise

# Instantiate Model
model = DiffusionModel(latent_dim=LATENT_DIM)
reverse_diffusion_model = ReverseDiffusion(model)

# Sample Random Data
x0 = torch.randn(1, LATENT_DIM)

# Sample Noise at Timestep t=10
t = 10
noisy_data, added_noise = forward_diffusion_sample(x0, t)

# Reverse Diffusion to Denoise Data
denoised_data = reverse_diffusion_model.reverse_diffusion(noisy_data, t)

# Print Sample Outputs
print("Original Data: ", x0)
print("Noisy Data: ", noisy_data)
print("Denoised Data: ", denoised_data)
