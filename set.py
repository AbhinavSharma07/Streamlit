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

# Control Set parameters
control_set = {
    "num_samples": 5,         # Number of images to generate
    "guidance_scale": 7.5,    # Strength of adherence to the prompt
    "learning_rate": 0.05,    # Latent space optimization learning rate
    "num_iterations": 15,     # Number of optimization iterations
    "diversity_factor": 0.5   # Factor controlling diversity vs precision
}

# Function to control diversity vs precision
def adjust_diversity(latents, diversity_factor):
    """
    Adjusts the diversity of the latent space by adding noise controlled by diversity_factor.
    """
    noise = torch.randn_like(latents) * diversity_factor
    return latents + noise

# Function to control the guided diffusion process with a control set
def guided_diffusion_with_control(prompt, control_set):
    num_samples = control_set["num_samples"]
    guidance_scale = control_set["guidance_scale"]
    learning_rate = control_set["learning_rate"]
    num_iterations = control_set["num_iterations"]
    diversity_factor = control_set["diversity_factor"]

    # Initialize random latent vectors
    latent_shape = (num_samples, pipe.unet.in_channels, 64, 64)  # latent size for stable diffusion
    latents = torch.randn(latent_shape).cuda()
    
    optimizer = Adam([latents], lr=learning_rate)
    
    for iteration in range(num_iterations):
        # Adjust latent space diversity
        latents = adjust_diversity(latents, diversity_factor)
        
        # Generate images from current latents
        images = []
        for i in range(num_samples):
            with torch.no_grad():
                latent_input = latents[i:i+1]
                image = pipe.decode_latents(latent_input)
                images.append(pipe.numpy_to_pil(image)[0])
        
        # Calculate CLIP loss and optimize
        total_loss = 0
        for i, image in enumerate(images):
            loss = calculate_clip_loss(image, prompt)
            total_loss += loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Log progress
        print(f"Iteration {iteration+1}/{num_iterations}, Loss: {total_loss.item():.4f}")
    
    # Final images after optimization
    final_images = []
    for i in range(num_samples):
        final_latent = latents[i:i+1]
        final_image = pipe.decode_latents(final_latent)
        final_images.append(pipe.numpy_to_pil(final_image)[0])
    
    return final_images

# Example Usage with Control Set
prompt = "A futuristic cityscape with flying cars and neon lights"
generated_images = guided_diffusion_with_control(prompt, control_set)

# Display or save the images
for idx, img in enumerate(generated_images):
    img.save(f"controlled_image_{idx+1}.png")
    img.show()
