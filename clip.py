''
import torch
from torchvision import datasets, transforms

# Data preprocessing (images normalization)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load dataset (replace with a suitable dataset for text-to-image tasks)
dataset = datasets.CIFAR10(root='data/', train=True, transform=transform, download=True)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)






import torch.nn as nn

# Generator
class Generator(nn.Module):
    def __init__(self, noise_dim, img_channels, img_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, img_size*img_size*img_channels),
            nn.Tanh()  # Output values between -1 and 1
        )

    def forward(self, z):
        img = self.model(z)
        return img.view(img.size(0), 3, 64, 64)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, img_size, img_channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_size*img_size*img_channels, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity





from transformers import CLIPProcessor, CLIPModel

# Load pre-trained CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Example: Encode text into latent space
def encode_text(prompt):
    inputs = clip_processor(text=[prompt], return_tensors="pt", padding=True)
    text_features = clip_model.get_text_features(**inputs)
    return text_features








  import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
img_size = 64
noise_dim = 100
img_channels = 3
learning_rate = 0.0002
epochs = 100

# Models
generator = Generator(noise_dim, img_channels, img_size).to(device)
discriminator = Discriminator(img_size, img_channels).to(device)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

# Loss function
adversarial_loss = nn.BCELoss()

for epoch in range(epochs):
    for i, (imgs, _) in enumerate(data_loader):
        # Training Discriminator
        valid = torch.ones(imgs.size(0), 1).to(device)
        fake = torch.zeros(imgs.size(0), 1).to(device)
        
        real_imgs = imgs.to(device)
        z = torch.randn(imgs.size(0), noise_dim).to(device)
        generated_imgs = generator(z)
        
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(generated_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # Training Generator
        g_loss = adversarial_loss(discriminator(generated_imgs), valid)
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch {epoch+1}/{epochs} | D Loss: {d_loss.item()} | G Loss: {g_loss.item()}")





import matplotlib.pyplot as plt
import numpy as np

def generate_image_from_text(prompt):
    # Encode text to latent representation
    text_features = encode_text(prompt).to(device)
    
    # Generate noise
    z = torch.randn(1, noise_dim).to(device)
    
    # Generate image
    with torch.no_grad():
        generated_img = generator(z).cpu().numpy()

    # Reshape and visualize
    generated_img = np.transpose(generated_img.squeeze(), (1, 2, 0))
    generated_img = 0.5 * (generated_img + 1)  # Rescale [-1, 1] to [0, 1]
    plt.imshow(generated_img)
    plt.show()

generate_image_from_text("a beautiful sunset over a mountain")



##############foundation for building #######################################################
