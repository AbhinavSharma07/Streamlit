"""
torch==2.0.0+cu118
torchvision==0.15.1+cu118
torchaudio==2.0.1+cu118
transformers==4.30.0
diffusers==0.19.0
scipy==1.11.0
numpy==1.23.5
Pillow==9.5.0
xformers==0.0.20
tqdm==4.65.0
opencv-python==4.7.0.72
matplotlib==3.7.2
huggingface-hub==0.14.1
omegaconf==2.3.0
"""

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

# Set up device for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained Stable Diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

# Generate an image from a text prompt
prompt = "A futuristic cityscape at sunset"
with autocast("cuda"):
    image = pipe(prompt).images[0]

# Save the generated image
image.save("futuristic_city.png")
