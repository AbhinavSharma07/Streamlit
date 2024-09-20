from diffusers import UNet2DConditionModel

class StableDiffusionModel:
    def __init__(self, model_id):
        # Load UNet Model for denoising
        self.unet = UNet2DConditionModel.from_pretrained(model_id)

    def denoise(self, latents, text_embedding, timestep):
        # Apply noise based on the UNet model
        with torch.no_grad():
            noise_pred = self.unet(latents, timestep, encoder_hidden_states=text_embedding).sample
        return noise_pred
