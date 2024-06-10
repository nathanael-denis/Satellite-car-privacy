import os
from inference_sdk import InferenceHTTPClient
import json
import PIL.Image
import PIL.ImageDraw
import torch
from diffusers import AutoPipelineForInpainting
import matplotlib.pyplot as plt

'''
This script merges several masks and apply inpainting only once using the same prompt,
which should be an issue as the cars can be located in different areas. 

Yet, the inpainting model is configured to be coherent with its surroundings, and the results is 
as good as inpainting each mask separately. Also, it is much faster and does not bear the risk 
of creating artifacts
'''
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com", 
    api_key="Your API Key on Roboflow"
)

init_image_path = r'C:\Code\Satellite-car-privacy-main\Google_Earth\car0001_512.png'
init_image = PIL.Image.open(init_image_path).convert("RGB")
new_image = init_image.resize((512, 512)) 
new_image.save('C:\Code\Satellite-car-privacy-main\Google_Earth\woomed_512.png')

result = CLIENT.infer(r'C:\Code\Satellite-car-privacy-main\Google_Earth\car0001_512.png', model_id="carsandswimmingpool/1")

print(result)

# Load the initial image from a local file
init_image_path = r'C:\Code\Satellite-car-privacy-main\Google_Earth\car0001_512.png'
init_image = PIL.Image.open(init_image_path).convert("RGB")

# Initialize the inpainting pipeline
pipeline = AutoPipelineForInpainting.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
)
pipeline.enable_model_cpu_offload()

# Define prompts
prompt = "a plain light grey, solid light grey, road"
negative_prompt = "colors, complex, diverse, mosaic, white"

# Create directories for masks and generated images if they don't exist
os.makedirs('masks', exist_ok=True)
os.makedirs('generatedImages', exist_ok=True)

# Create a combined mask
combined_mask = PIL.Image.new("L", init_image.size, 0)
draw = PIL.ImageDraw.Draw(combined_mask)

# Iterate over predictions and generate masks
for idx, prediction in enumerate(result['predictions']):
    print("Prediction :", prediction)
    # Increase bounding box size by 20%
    new_width = prediction['width'] * 2.5
    new_height = prediction['height'] * 2.5
    new_x = prediction['x']
    new_y = prediction['y']
    print("Width:", new_width, "Height:", new_height, "x:", new_x, 'y:', new_y)
    
    # Draw rectangle on the combined mask
    box = [
        new_x - new_width / 2,
        new_y - new_height / 2,
        new_x + new_width / 2,
        new_y + new_height / 2
    ]
    draw.rectangle(box, fill=255)

# Save the combined mask image
combined_mask_image_path = "masks/combined_mask_image.png"
combined_mask.save(combined_mask_image_path)

# Generate the inpainted image using the combined mask
image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=init_image, mask_image=combined_mask).images[0]

# Save the generated image
generated_image_path = "generatedImages/generated_image.png"
image.save(generated_image_path)

# Display the generated image
plt.figure()
plt.imshow(image)
plt.axis('off')  # Hide axes
plt.title('Generated Image')
plt.show()

print('-------------------Inpainting done--------------')