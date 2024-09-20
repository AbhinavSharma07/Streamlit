import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
from torch.optim import Adam
import numpy as np

# Load Stable Diffusion and CLIP models
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to("cuda")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Function to calculate CLIP similarity loss
def calculate_clip_loss(image, prompt):
    inputs = clip_processor(text=[prompt], images=image, return_tensors="pt").to("cuda")
    outputs = clip_model(**inputs)
    return -outputs.logits_per_image

# Enhanced Latent Space Exposure function
def enhanced_clip_guided_diffusion(prompt, num_inference_steps=50, guidance_scale=7.5, num_iterations=10, num_samples=3):
    """
    Generate multiple images by traversing different regions of the latent space.
    """
    # Initialize different latent vectors to explore different parts of the latent space
    latent_shape = (num_samples, pipe.unet.in_channels, 64, 64)  # latent size for stable diffusion
    latents = torch.randn(latent_shape).cuda()  # Start with random noise
    
    optimizer = Adam([latents], lr=0.05)
    clip_losses = []
    
    # Perform optimization in latent space
    for iteration in range(num_iterations):
        images = []
        for i in range(num_samples):
            with torch.no_grad():
                latent_input = latents[i:i+1]
                image = pipe.decode_latents(latent_input)
                images.append(pipe.numpy_to_pil(image)[0])
        
        total_loss = 0
        for i, image in enumerate(images):
            loss = calculate_clip_loss(image, prompt)
            total_loss += loss
            clip_losses.append(loss.item())
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Log progress
        print(f"Iteration {iteration+1}/{num_iterations}, Loss: {total_loss.item():.4f}")
    
    # Final images after exposure to different latent space regions
    final_images = []
    for i in range(num_samples):
        final_latent = latents[i:i+1]
        final_image = pipe.decode_latents(final_latent)
        final_images.append(pipe.numpy_to_pil(final_image)[0])
    
    return final_images

# Example Usage
prompt = "A futuristic cityscape with flying cars and neon lights"
generated_images = enhanced_clip_guided_diffusion(prompt, num_samples=5, num_iterations=15)

# Display or save the images
for idx, img in enumerate(generated_images):
    img.save(f"generated_image_{idx+1}.png")
    img.show()
