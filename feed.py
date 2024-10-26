\\\
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch.optim as optim

# Load the Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe.to("cuda")

# Load the pre-trained CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Function to calculate CLIP similarity between image and text
def calculate_clip_loss(image, prompt):
    """
    Calculate the CLIP loss (negative similarity score between image and text).
    A lower value indicates higher alignment between the image and text.
    """
    inputs = clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True).to("cuda")
    outputs = clip_model(**inputs)
    
    # Return the negative similarity score as the loss (lower is better)
    return -outputs.logits_per_image

# Function to perform gradient update on the latent space
def clip_guided_diffusion(prompt, num_inference_steps=50, guidance_scale=7.5, num_iterations=5, learning_rate=0.05):
    """
    Perform CLIP-guided image generation with backpropagation to adjust latent vectors.
    
    Args:
        prompt (str): Input text for generating an image.
        num_inference_steps (int): Number of diffusion steps.
        guidance_scale (float): How strongly the model focuses on the prompt.
        num_iterations (int): Number of CLIP-guided iterations.
        learning_rate (float): Learning rate for gradient-based latent space updates.

    Returns:
        final_image (PIL.Image): The final CLIP-guided image.
    """
    # Initial image generation (without CLIP guidance yet)
    latents = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, output_type="latent").latent
    
    # Set up optimizer for adjusting latent vectors
    latents = latents.requires_grad_(True)  # Allow gradients for latents
    optimizer = optim.Adam([latents], lr=learning_rate)

    # Iterate through CLIP-guided refinement steps
    for i in range(num_iterations):
        # Generate image from current latents
        image = pipe.decode_latents(latents)
        image = pipe.numpy_to_pil(image)[0]

        # Calculate CLIP loss
        clip_loss = calculate_clip_loss(image, prompt)
        
        # Backpropagate through CLIP loss to adjust latents
        optimizer.zero_grad()
        clip_loss.backward()
        optimizer.step()

        print(f"Iteration {i + 1}/{num_iterations}, CLIP loss: {clip_loss.item():.4f}")

    # Final image generation after all CLIP-guided updates
    final_image = pipe.decode_latents(latents)
    final_image = pipe.numpy_to_pil(final_image)[0]
    
    return final_image

# Example usage
prompt = "A painting of a futuristic city skyline at sunset"
final_image = clip_guided_diffusion(prompt, num_iterations=10)

# Save or show the final image
final_image.save("final_clip_guided_image.png")
final_image.show()
