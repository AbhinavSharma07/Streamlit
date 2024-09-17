import os
import torch
import asyncio
import logging
import argparse
from PIL import Image, ImageFilter, ImageEnhance
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the Stable Diffusion pipeline
def initialize_pipeline(model_id="CompVis/stable-diffusion-v1-4", device="cuda", custom_config=None):
    """Initialize the Stable Diffusion pipeline with optional custom configuration."""
    try:
        logging.info("Initializing Stable Diffusion pipeline...")
        pipe = StableDiffusionPipeline.from_pretrained(model_id)
        if custom_config:
            logging.info("Applying custom configuration...")
            pipe.scheduler = custom_config.get('scheduler', pipe.scheduler)
            pipe.vae = custom_config.get('vae', pipe.vae)
        pipe = pipe.to(device)
        logging.info("Pipeline initialized successfully.")
        return pipe
    except Exception as e:
        logging.error(f"Error initializing pipeline: {e}")
        raise

# Asynchronous image generation
async def generate_image_async(pipe, prompt, num_inference_steps=50, guidance_scale=7.5, generator=None):
    """Generate a single image asynchronously."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator).images[0])

async def generate_images_async(pipe, prompts, num_inference_steps=50, guidance_scale=7.5, seed=None):
    """Generate multiple images asynchronously."""
    images = []
    generator = torch.manual_seed(seed) if seed is not None else None
    
    tasks = [generate_image_async(pipe, prompt, num_inference_steps, guidance_scale, generator) for prompt in prompts]
    for i, task in enumerate(asyncio.as_completed(tasks)):
        try:
            logging.info(f"Generating image {i+1}/{len(prompts)}...")
            image = await task
            images.append(image)
        except Exception as e:
            logging.error(f"Error generating image for prompt '{prompts[i]}': {e}")
            images.append(None)
    
    return images

# Display images using matplotlib
def display_images(images, titles=None):
    """Display generated images using matplotlib."""
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    
    if num_images == 1:
        axes = [axes]
    
    for ax, img, title in zip(axes, images, titles if titles else [None] * num_images):
        if img:
            ax.imshow(img)
            ax.axis("off")
            if title:
                ax.set_title(title, fontsize=10)
        else:
            ax.text(0.5, 0.5, "Error", fontsize=12, ha='center')
            ax.axis("off")
    plt.show()

# Save images with optional postprocessing
def save_images(images, output_dir="generated_images", postprocess=None):
    """Save generated images to disk with optional postprocessing."""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, img in enumerate(images):
        if img:
            if postprocess:
                img = apply_postprocessing(img, postprocess)
            img_path = os.path.join(output_dir, f"image_{i+1}.png")
            img.save(img_path)
            logging.info(f"Image saved to {img_path}")
        else:
            logging.warning(f"Skipping save for image {i+1} due to generation error.")

# Apply postprocessing to an image
def apply_postprocessing(img, postprocess):
    """Apply postprocessing techniques to an image."""
    if 'filter' in postprocess:
        filter_type = postprocess['filter']
        if filter_type == 'BLUR':
            img = img.filter(ImageFilter.BLUR)
        elif filter_type == 'SHARPEN':
            img = img.filter(ImageFilter.SHARPEN)
        elif filter_type == 'CONTOUR':
            img = img.filter(ImageFilter.CONTOUR)
    
    if 'enhance' in postprocess:
        factor = postprocess.get('enhance_factor', 1.5)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(factor)
    
    return img

# Custom image preprocessing
def preprocess_image(image, preprocess_options):
    """Preprocess image before passing it to the pipeline."""
    if 'resize' in preprocess_options:
        size = preprocess_options['resize']
        image = image.resize(size, Image.ANTIALIAS)
    if 'crop' in preprocess_options:
        box = preprocess_options['crop']
        image = image.crop(box)
    return image

# Handle multiple threads for image generation
def threaded_generate_images(pipe, prompts, num_inference_steps=50, guidance_scale=7.5, seed=None):
    """Generate images using multiple threads."""
    images = []
    generator = torch.manual_seed(seed) if seed is not None else None
    
    def generate(prompt):
        try:
            return pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator).images[0]
        except Exception as e:
            logging.error(f"Error generating image for prompt '{prompt}': {e}")
            return None
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(generate, prompt) for prompt in prompts]
        for i, future in enumerate(futures):
            try:
                logging.info(f"Generating image {i+1}/{len(prompts)}...")
                image = future.result()
                images.append(image)
            except Exception as e:
                logging.error(f"Error generating image for prompt '{prompts[i]}': {e}")
                images.append(None)
    
    return images

# Parse command-line arguments with extended options
def parse_args():
    """Parse command-line arguments for image generation."""
    parser = argparse.ArgumentParser(description="Generate images using Stable Diffusion")
    parser.add_argument('--model_id', type=str, default="CompVis/stable-diffusion-v1-4", help="Model ID for Stable Diffusion")
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default="cuda", help="Device to use for computation")
    parser.add_argument('--prompts', type=str, nargs='+', required=True, help="List of prompts for image generation")
    parser.add_argument('--num_inference_steps', type=int, default=50, help="Number of inference steps")
    parser.add_argument('--guidance_scale', type=float, default=7.5, help="Guidance scale")
    parser.add_argument('--seed', type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument('--output_dir', type=str, default="generated_images", help="Directory to save generated images")
    parser.add_argument('--postprocess_filter', type=str, choices=['BLUR', 'SHARPEN', 'CONTOUR'], help="Postprocessing filter")
    parser.add_argument('--postprocess_enhance', action='store_true', help="Apply contrast enhancement")
    parser.add_argument('--enhance_factor', type=float, default=1.5, help="Factor for contrast enhancement")
    parser.add_argument('--preprocess_resize', type=int, nargs=2, help="Resize image to width and height (e.g., 256 256)")
    parser.add_argument('--preprocess_crop', type=int, nargs=4, help="Crop image with left, upper, right, lower (e.g., 10 20 100 200)")
    parser.add_argument('--async', action='store_true', help="Use asynchronous image generation")
    return parser.parse_args()

# Main function
def main():
    args = parse_args()
    
    # Initialize pipeline
    device = args.device
    custom_config = {
        'scheduler': None,  # Example placeholder, customize if needed
        'vae': None  # Example placeholder, customize if needed
    }
    pipe = initialize_pipeline(model_id=args.model_id, device=device, custom_config=custom_config)
    
    # Preprocess options
    preprocess_options = {}
    if args.preprocess_resize:
        preprocess_options['resize'] = tuple(args.preprocess_resize)
    if args.preprocess_crop:
        preprocess_options['crop'] = tuple(args.preprocess_crop)
    
    # Generate images
    prompts = args.prompts
    postprocess = {
        'filter': args.postprocess_filter,
        'enhance': args.postprocess_enhance,
        'enhance_factor': args.enhance_factor
    }
    
    # Apply preprocessing if specified
    if preprocess_options:
        prompts = [preprocess_image(Image.new("RGB", (256, 256)), preprocess_options) for _ in prompts]
    
    if args.async:
        loop = asyncio.get_event_loop()
        images = loop.run_until_complete(generate_images_async(pipe, prompts, num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale, seed=args.seed))
    else:
        images = threaded_generate_images(pipe, prompts, num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale, seed=args.seed)
    
    # Display images
    display_images(images, titles=prompts)
    
    # Save images
    save_images(images, output_dir=args.output_dir, postprocess=postprocess)

# Unit tests
def test_functions():
    """Unit tests for the functions."""
    # Example test cases, customize according to your testing needs
    
    # Test initialization
    try:
        pipe = initialize_pipeline()
        assert pipe is not None
        logging.info("Initialization test passed.")
    except AssertionError:
        logging.error("Initialization test failed.")
    
    # Test image generation
    try:
        pipe = initialize_pipeline()
        images = threaded_generate_images(pipe, ["A serene beach"], num_inference_steps=10, guidance_scale=5.0)
        assert len(images) == 1
        logging.info("Image generation test passed.")
    except AssertionError:
        logging.error("Image generation test failed.")
    
    # Test image processing
    try:
        img = Image.new("RGB", (256, 256))
        processed_img = apply_postprocessing(img, {'filter': 'SHARPEN'})
        assert processed_img is not None
        logging.info("Postprocessing test passed.")
    except AssertionError:
        logging.error("Postprocessing test failed.")

if __name__ == "__main__":
    main()
    test_functions()






#####USAGE#######################





python script_name.py --prompts "A serene beach at sunset" "A futuristic cityscape with flying cars" --num_inference_steps 100 --guidance_scale 8.0 --seed 42 --output_dir my_images --postprocess_filter SHARPEN --postprocess_enhance --enhance_factor 2.0 --preprocess_resize 256 256 --preprocess_crop 10 20 100 200 --async
