//
import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import matplotlib.pyplot as plt

# Initialize the Stable Diffusion pipeline
def initialize_pipeline(model_id="CompVis/stable-diffusion-v1-4", device="cuda"):
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe = pipe.to(device)
    return pipe

# Generate an image from a text prompt
def generate_image(pipe, prompt, num_inference_steps=50, guidance_scale=7.5, seed=None):
    if seed is not None:
        generator = torch.manual_seed(seed)
    else:
        generator = None

    # Generate image
    image = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator).images[0]
    return image

# Display the image using matplotlib
def display_image(image):
    plt.imshow(image)
    plt.axis("off")
    plt.show()

# Main function
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt = "A fantasy landscape with mountains and a river"
    seed = 42  # Optional seed for reproducibility
    
    # Initialize pipeline
    pipe = initialize_pipeline(device=device)
    
    # Generate image
    image = generate_image(pipe, prompt, seed=seed)
    
    # Display image
    display_image(image)

if __name__ == "__main__":
    main()
