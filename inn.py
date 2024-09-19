from diffusers import StableDiffusionPipeline
import torch

# Load the Stable Diffusion pipeline
model_id = "CompVis/stable-diffusion-v1-4"  # You can use other models as well
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")  # Use CUDA if available

# Define the prompt for generating an image
prompt = "A scenic view of mountains during sunset"

# Generate an image based on the prompt
image = pipe(prompt).images[0]

# Save the generated image
image.save("generated_image.png")
