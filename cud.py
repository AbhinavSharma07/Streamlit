import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import matplotlib.pyplot as plt 

# Initialize the Stable Diffusion pipeline
def initialize_pipeline(model_id="CompVis/stable-diffusion-v1-4", device="cuda"):
    try:
        pipe = StableDiffusionPipeline.from_pretrained(model_id)
        pipe = pipe.to(device)
        print("Pipeline initialized successfully.")
        return pipe
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        raise

# Generate images from a list of prompts
def generate_images(pipe, prompts, num_inference_steps=50, guidance_scale=7.5, seed=None):
    images = []
    generator = torch.manual_seed(seed) if seed is not None else None
    
    for prompt in prompts:
        try:
            print(f"Generating image for prompt: {prompt}")
            image = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator).images[0]
            images.append(image)
        except Exception as e:
            print(f"Error generating image for prompt '{prompt}': {e}")
            images.append(None)
    
    return images

# Display images using matplotlib
def display_images(images):
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    for ax, img in zip(axes, images):
        if img:
            ax.imshow(img)
            ax.axis("off")
        else:
            ax.text(0.5, 0.5, "Error", fontsize=12, ha='center')
            ax.axis("off")
    plt.show()

# Save images to disk
def save_images(images, output_dir="generated_images"):
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(images):
        if img:
            img_path = os.path.join(output_dir, f"image_{i}.png")
            img.save(img_path)
            print(f"Image saved to {img_path}")
        else:
            print(f"Skipping save for image {i} due to generation error.")

# Main function
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompts = [
        "A serene beach at sunset",
        "A futuristic cityscape with flying cars",
        "A cozy cabin in the snowy mountains"
    ]
    num_inference_steps = 50
    guidance_scale = 7.5
    seed = 42  # Optional seed for reproducibility

    # Initialize pipeline
    pipe = initialize_pipeline(device=device)
    
    # Generate images
    images = generate_images(pipe, prompts, num_inference_steps, guidance_scale, seed)
    
    # Display images
    display_images(images)
    
    # Save images
    save_images(images)

if __name__ == "__main__":
    main()
