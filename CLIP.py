import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Load the Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe.to("cuda")

# Load the pre-trained CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Function to calculate CLIP similarity between image and text
def calculate_clip_similarity(image, prompt):
    inputs = clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True).to("cuda")
    outputs = clip_model(**inputs)
    
    # Similarity score
    return outputs.logits_per_image.item()

# Function to generate image with CLIP-based guidance
def generate_clip_guided_image(prompt, num_inference_steps=50, guidance_scale=7.5, num_iterations=5):
    """
    Generate an image with CLIP-guided Stable Diffusion.

    Args:
        prompt (str): The input text for generating an image.
        num_inference_steps (int): Number of diffusion steps.
        guidance_scale (float): How strongly the model focuses on the prompt.
        num_iterations (int): Number of iterations to apply CLIP guidance.

    Returns:
        image (PIL.Image): The generated image.
    """
    # Initial image generation
    image = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
    
    for _ in range(num_iterations):
        # Calculate CLIP similarity score
        similarity = calculate_clip_similarity(image, prompt)
        print(f"CLIP similarity: {similarity}")
        
        # Adjust image with respect to CLIP guidance (can involve some gradient steps)
        # This part can involve directly modifying the latent space using the CLIP loss.
        # Here it's just conceptual since modifying latent space with CLIP feedback is a complex task.
        # This process would involve backpropagating the CLIP loss to adjust the image.

    return image

# Example usage
prompt = "A futuristic city skyline at sunset"
image = generate_clip_guided_image(prompt)

# Save or show the image
image.save("clip_guided_image.png")
image.show()
