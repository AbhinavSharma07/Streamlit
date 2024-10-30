;''
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import os

# Function to generate images using Stable Diffusion
def generate_images(prompt, num_images=1, guidance_scale=7.5, num_inference_steps=50, height=512, width=512, save_path="generated_images"):
    """
    Generates images using the Stable Diffusion model.
    
    Args:
        prompt (str): The prompt for generating images.
        num_images (int): Number of images to generate.
        guidance_scale (float): How much the model should follow the prompt (higher = more strict).
        num_inference_steps (int): The number of inference steps (higher = better quality, but slower).
        height (int): Height of the generated images.
        width (int): Width of the generated images.
        save_path (str): Directory to save the generated images.

    Returns:
        None
    """
    # Load the Stable Diffusion model
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")  # Move the model to GPU

    # Create the save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    print(f"Generating {num_images} image(s) based on prompt: '{prompt}'")
    
    # Generate multiple images
    for i in range(num_images):
        print(f"Generating image {i + 1} of {num_images}...")
        image = pipe(prompt, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, height=height, width=width).images[0]

        # Save the image
        image_path = os.path.join(save_path, f"image_{i + 1}.png")
        image.save(image_path)
        print(f"Image {i + 1} saved to: {image_path}")

# Example Usage
if __name__ == "__main__":
    # Define your prompt
    prompt = "A futuristic cityscape with flying cars and tall glass buildings, during sunset, highly detailed, vibrant colors"

    # Generate 5 images with custom parameters
    generate_images(prompt, num_images=5, guidance_scale=8.0, num_inference_steps=100, height=768, width=768)
