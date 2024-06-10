import os
from inference_sdk import InferenceHTTPClient
import json
import PIL.Image
import PIL.ImageDraw
import requests
import cv2
import torch
from io import BytesIO
from diffusers import AutoPipelineForInpainting
from diffusers.utils import make_image_grid
import matplotlib.pyplot as plt

'''
This script try to apply inpainting on an image with more than 5 cars, close to one another, which is a challenging task for the inpainting model. When looping, the image starts to seriously degrade after a few steps, and the cars are not properly removed.

'''
 
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="Your API Key on Roboflow"
)

init_image_path = r'C:\Code\Satellite-car-privacy-main\Google_Earth\woomed.png'
init_image = PIL.Image.open(init_image_path).convert("RGB")
new_image = init_image.resize((512, 512)) 
new_image.save('C:\Code\Satellite-car-privacy-main\Google_Earth\woomed_512.png')

result = CLIENT.infer(r'C:\Code\Satellite-car-privacy-main\Google_Earth\woomed_512.png', model_id="carsandswimmingpool/1")
print(result)

def parse_json(data):
    # Start parsing
    output = []
    output.append(f"Time: {data['time']} seconds")
    output.append(f"Image Dimensions: {data['image']['width']}x{data['image']['height']}")
    
    output.append("Predictions:")
    for prediction in data['predictions']:
        pred_details = (
            f"  Detection ID: {prediction['detection_id']}\n"
            f"    Class ID: {prediction['class_id']} ({prediction['class']})\n"
            f"    Bounding Box: [x: {prediction['x']}, y: {prediction['y']}, width: {prediction['width']}, height: {prediction['height']}]\n"
            f"    Confidence: {prediction['confidence']:.2f}\n"
        )
        output.append(pred_details)
    
    return "\n".join(output)

print('-------------------Classification done--------------')

# Load the initial image from a local file
init_image_path = r'C:\Code\Satellite-car-privacy-main\Google_Earth\woomed_512.png'
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

# Iterate over predictions and generate masks
for idx, prediction in enumerate(result['predictions']):
    print("Prediction :",prediction)
    # Increase bounding box size by 20%
    new_width = prediction['width'] * 2.5
    new_height = prediction['height'] * 2.5
    new_x = prediction['x']
    new_y = prediction['y']
    print("Width:",new_width,"Height:", new_height,"x:", new_x,'y:', new_y)
    # Create a mask image based on the prediction's bounding box
    mask_image = PIL.Image.new("L", init_image.size, 0)
    draw = PIL.ImageDraw.Draw(mask_image)
    
    box = [
        new_x - new_width / 2,
        new_y - new_height / 2,
        new_x + new_width / 2,
        new_y + new_height / 2
    ]
    draw.rectangle(box, fill=255)
    
    # Save the mask image
    mask_image_path = f"masks/mask_image_{idx}.png"
    mask_image.save(mask_image_path)
    
    # Generate the inpainted image
    image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=init_image, mask_image=mask_image).images[0]

    # Save the generated image
    generated_image_path = f"generatedImages/generated_image_{idx}.png"
    image.save(generated_image_path)
    
    # Use the newly generated image as input for the next iteration
    init_image = image
    
    # Display the generated image
    plt.figure()
    plt.imshow(image)
    plt.axis('off')  # Hide axes
    plt.title(f'Generated Image {idx + 1}')
    plt.show(block=False)
    plt.pause(1)  # Pause for a second to display the image

# Close all figures
plt.close('all')

print('-------------------Inpainting done--------------')
