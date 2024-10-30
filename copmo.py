
import torch
from torch import autocast
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, LMSDiscreteScheduler
from PIL import Image

class StableDiffusionModel:
    def __init__(self, model_id):
        # Load the components
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
        self.text_encoder = CLIPTextModel.from_pretrained(model_id)
        self.unet = UNet2DConditionModel.from_pretrained(model_id)
        self.scheduler = LMSDiscreteScheduler.from_pretrained(model_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def encode_text(self, prompt):
        # Encode the text prompt
        text_input = self.tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
        with torch.no_grad():
            text_embedding = self.text_encoder(text_input.input_ids.to(self.device))[0]
        return text_embedding

    def sample_latent_noise(self, batch_size, latent_shape):
        # Sample random noise
        return torch.randn((batch_size, *latent_shape)).to(self.device)

    def denoise(self, latents, text_embedding, timestep):
        # Predict noise with UNet
        with torch.no_grad():
            noise_pred = self.unet(latents, timestep, encoder_hidden_states=text_embedding).sample
        return noise_pred

    def step_scheduler(self, noise_pred, timestep, latents):
        # Update latents based on the noise prediction
        latents = self.scheduler.step(noise_pred, timestep, latents).prev_sample
        return latents

    def decode_latents(self, latents):
        # Decode the latents to an image
        latents = latents.cpu().detach().numpy()
        image = ((latents + 1.0) * 127.5).astype("uint8")
        return Image.fromarray(image)

    def generate_image(self, prompt, num_steps=50, guidance_scale=7.5):
        # Main image generation loop
        text_embedding = self.encode_text(prompt)
        latents = self.sample_latent_noise(batch_size=1, latent_shape=(4, 64, 64))
        for timestep in self.scheduler.timesteps:
            with autocast("cuda"):
                noise_pred = self.denoise(latents, text_embedding, timestep)
                noise_pred = noise_pred * guidance_scale
                latents = self.step_scheduler(noise_pred, timestep, latents)
        image = self.decode_latents(latents)
        return image

# Example usage:
model = StableDiffusionModel(model_id="CompVis/stable-diffusion-v1-4")
image = model.generate_image("A sunset over a futuristic city")
image.save("generated_image.png")
