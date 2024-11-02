'''''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from transformers import CLIPModel, CLIPProcessor
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import random


# Transformations for the image data
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])

# Sample dataset (You can use a custom paired text-image dataset)
dataset = datasets.CIFAR10(root='./data', download=True, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Load CLIP Model for Text Encoding
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def encode_text(prompt):
    inputs = clip_processor(text=[prompt], return_tensors="pt", padding=True)
    text_features = clip_model.get_text_features(**inputs)
    return text_features


class Generator(nn.Module):
    def __init__(self, noise_dim, text_dim, img_channels, img_size):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(noise_dim + text_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, img_size * img_size * img_channels),
            nn.Tanh()  # Output is scaled to [-1, 1]
        )

    def forward(self, z, text_embedding):
        input_vec = torch.cat((z, text_embedding), dim=1)
        img = self.fc(input_vec)
        return img.view(img.size(0), 3, 128, 128)


class Discriminator(nn.Module):
    def __init__(self, img_size, img_channels, text_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(img_size * img_size * img_channels + text_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output is probability of real or fake
        )

    def forward(self, img, text_embedding):
        img_flat = img.view(img.size(0), -1)
        input_vec = torch.cat((img_flat, text_embedding), dim=1)
        validity = self.fc(input_vec)
        return validity




class Attention(nn.Module):
    def __init__(self, text_dim, attention_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(text_dim, attention_dim)

    def forward(self, text_embedding):
        attention_weights = torch.softmax(self.attention(text_embedding), dim=1)
        attended_text = attention_weights * text_embedding
        return attended_text


def clip_loss(generated_img, text_embedding):
    clip_image_features = clip_model.get_image_features(pixel_values=generated_img)
    loss = 1 - torch.cosine_similarity(clip_image_features, text_embedding)
    return loss.mean()

# Adversarial loss (GAN Loss)
adversarial_loss = nn.BCELoss()

# Function to calculate combined loss
def combined_loss(discriminator, real_imgs, generated_imgs, text_embedding):
    # Real images
    valid = torch.ones(real_imgs.size(0), 1).to(device)
    fake = torch.zeros(real_imgs.size(0), 1).to(device)

    real_loss = adversarial_loss(discriminator(real_imgs, text_embedding), valid)
    fake_loss = adversarial_loss(discriminator(generated_imgs.detach(), text_embedding), fake)
    d_loss = (real_loss + fake_loss) / 2

    # Generator loss with adversarial + CLIP guidance
    g_loss = adversarial_loss(discriminator(generated_imgs, text_embedding), valid)
    g_loss += clip_loss(generated_imgs, text_embedding)

    return d_loss, g_loss



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
noise_dim = 100
text_dim = 512
img_size = 128
img_channels = 3

# Initialize models
generator = Generator(noise_dim, text_dim, img_channels, img_size).to(device)
discriminator = Discriminator(img_size, img_channels, text_dim).to(device)
attention = Attention(text_dim, text_dim).to(device)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training Loop
epochs = 100

for epoch in range(epochs):
    for i, (imgs, _) in enumerate(data_loader):
        real_imgs = imgs.to(device)

        # Generate text embeddings for a prompt
        prompt = "a dog playing in the park"
        text_embedding = encode_text(prompt).to(device)
        text_embedding = attention(text_embedding)

        # Generate fake images
        z = torch.randn(real_imgs.size(0), noise_dim).to(device)
        generated_imgs = generator(z, text_embedding)

        # Loss calculation
        d_loss, g_loss = combined_loss(discriminator, real_imgs, generated_imgs, text_embedding)

        # Update Discriminator
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # Update Generator
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch+1}/{epochs}] | D Loss: {d_loss.item()} | G Loss: {g_loss.item()}")


def generate_image_from_text(prompt):
    text_embedding = encode_text(prompt).to(device)
    text_embedding = attention(text_embedding)
    z = torch.randn(1, noise_dim).to(device)
    generated_img = generator(z, text_embedding).cpu().detach().numpy()

    # Rescale image to [0, 1] and display
    generated_img = 0.5 * (generated_img + 1)
    generated_img = np.transpose(generated_img.squeeze(), (1, 2, 0))
    plt.imshow(generated_img)
    plt.show()

generate_image_from_text("a beautiful sunset over the mountains")




####################################PUNISHER##################################################
