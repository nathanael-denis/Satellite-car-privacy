import os
from inference_sdk import InferenceHTTPClient
import PIL.Image
import PIL.ImageDraw
import torch
from diffusers import AutoPipelineForInpainting
import matplotlib.pyplot as plt

'''
Entering main part of the project, takes a directory holding several satellite images and cleans them by removing the cars.

The images used are the ones used to validate the model, should removed 97% of the car ( model accuracy) but inpainting is 
also not 100% accurate and may not erase the car properly. Also, inpainting has some distortion effect on some pictures, that are 
probably related to the training of the inpainting model.
'''

# Initialize the inference client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="Your API Key on Roboflow"
)

# Directory paths
input_dir = r'C:\Code\Satellite-car-privacy-main\databaseLowRes'
output_dir = r'C:\Code\Satellite-car-privacy-main\generatedImages'
mask_dir = os.path.join(output_dir, 'masks')
generated_dir = os.path.join(output_dir, 'generatedImages')

# Create directories if they don't exist
os.makedirs(mask_dir, exist_ok=True)
os.makedirs(generated_dir, exist_ok=True)

# Initialize the inpainting pipeline
pipeline = AutoPipelineForInpainting.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
)
pipeline.enable_model_cpu_offload()

# Define prompts
prompt = "a plain light grey, solid light grey, road"
negative_prompt = "colors, complex, diverse, mosaic, white"

# Process each image in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
        image_path = os.path.join(input_dir, filename)
        print(f'Processing {image_path}')
        
        # Load the initial image
        init_image = PIL.Image.open(image_path).convert("RGB")

        # Resize the image
        new_image = init_image.resize((512, 512))
        resized_image_path = os.path.join(output_dir, f'resized_{filename}')
        #new_image.save(resized_image_path)

        # Run inference to detect objects
        result = CLIENT.infer(image_path, model_id="carsandswimmingpool/1")
        print(result)

        # Create a combined mask
        combined_mask = PIL.Image.new("L", init_image.size, 0)
        draw = PIL.ImageDraw.Draw(combined_mask)

        # Iterate over predictions and generate the mask
        for idx, prediction in enumerate(result['predictions']):
            print("Prediction :", prediction)
            # Increase bounding box sizes
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
        combined_mask_image_path = os.path.join(mask_dir, f'mask_{filename}')
        combined_mask.save(combined_mask_image_path)

        # Generate the inpainted image using the combined mask
        image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=init_image, mask_image=combined_mask).images[0]

        # Save the generated image
        generated_image_path = os.path.join(generated_dir, f'generated_{filename}')
        image.save(generated_image_path)

        # Display the generated image (optional)
        plt.figure()
        plt.imshow(image)
        plt.axis('off')  # Hide axes
        plt.title(f'Generated Image: {filename}')
        #plt.show()

        print('-------------------Inpainting done for', filename, '--------------')

print('-------------------All images processed--------------')