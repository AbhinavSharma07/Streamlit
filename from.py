from transformers import CLIPTextModel, CLIPTokenizer

class StableDiffusionModel:
    def __init__(self, model_id):
        # Load the CLIP model and tokenizer for text encoding
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
        self.text_encoder = CLIPTextModel.from_pretrained(model_id)

    def encode_text(self, prompt):
        # Tokenize the text and convert it to latent features
        text_input = self.tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
        with torch.no_grad():
            text_embedding = self.text_encoder(text_input.input_ids)[0]
        return text_embedding
