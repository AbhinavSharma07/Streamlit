

for time_step in reversed(range(num_steps)):
  
    noise = sample_noise(image_shape)
    
 
    conditional_noise = model(noise, text_encoding, time_step)
    
   
    generated_image = denoise(generated_image, conditional_noise, time_step)


adversarial_loss = adversarial_loss(discriminator(generated_image), real_labels)
clip_loss = 1 - cosine_similarity(clip_image_embedding, clip_text_embedding)
total_loss = adversarial_loss + lambda_clip * clip_loss



for i in range(num_images):
    z = torch.randn(1, noise_dim).to(device)
    generated_img = generator(z).cpu().detach().numpy()


