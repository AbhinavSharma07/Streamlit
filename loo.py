import torch
from torch import autocast
from PIL import Image

class StableDiffusionModel:
    def generate_image(self, prompt, num_steps=50, guidance_scale=7.5):
        # Encode the text into latent space
        text_embedding = self.encode_text(prompt)

        # Sample random noise as the initial latent image
        latents = self.sample_latent_noise(batch_size=1, latent_shape=(4, 64, 64))
        latents = latents.to(device)

        # Loop over timesteps to gradually refine the image
        for timestep in self.scheduler.timesteps:
            with autocast("cuda"):
                # Denoise step
                noise_pred = self.denoise(latents, text_embedding, timestep)
                
                # Guidance (scale the predicted noise)
                noise_pred = noise_pred * guidance_scale
                
                # Update latent image (subtract noise)
                latents = self.step_scheduler(noise_pred, timestep, latents)

        # Decode latents to pixel space (e.g., using VAE or other decoder)
        image = self.decode_latents(latents)
        return image

    def decode_latents(self, latents):
        # Decode latent space to an image
        latents = latents.cpu().detach().numpy()
        image = ((latents + 1.0) * 127.5).astype("uint8")
        return Image.fromarray(image)
