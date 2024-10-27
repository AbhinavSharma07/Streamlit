\\\
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image

# Ensure we have a GPU to run stable diffusion
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the stable diffusion model from HuggingFace
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe.to(device)

# Set scheduler (optional but recommended for speed)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Function to generate an image
def generate_image(prompt, num_inference_steps=50, guidance_scale=7.5, height=512, width=512):
    """
    Generate an image from a text prompt using the Stable Diffusion model.

    Args:
        prompt (str): The input text for generating an image.
        num_inference_steps (int): The number of diffusion steps (higher is slower but better quality).
        guidance_scale (float): Controls how much the model focuses on the text prompt.
        height (int): The height of the generated image.
        width (int): The width of the generated image.

    Returns:
        image (PIL Image): The generated image.
    """
    image = pipe(
        prompt, 
        num_inference_steps=num_inference_steps, 
        guidance_scale=guidance_scale, 
        height=height, 
        width=width
    ).images[0]
    return image

# Example usage
prompt = "A futuristic city skyline at sunset"
image = generate_image(prompt)

# Save or show the image
image.save("generated_image.png")
image.show()








from transformers import CLIPTextModel, CLIPTokenizer

# Load pre-trained CLIP model and tokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

# Tokenize input prompt
prompt = "A scenic view of mountains during sunset"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text embeddings
text_embeddings = text_encoder(**inputs).last_hidden_state






# Sample random noise as a starting point for diffusion
latent = torch.randn((1, 4, 64, 64)).to(device)

# Iterate through diffusion steps to denoise
for i in range(num_inference_steps):
    noise_pred = model(latent, text_embeddings, timestep=i)
    latent = scheduler.step(noise_pred, latent, timestep=i)






# Decode the latent representation back to an image
decoded_image = vae.decode(latent)
