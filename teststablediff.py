# import torch
# from diffusers import AutoPipelineForInpainting
# from diffusers.utils import load_image, make_image_grid
# import torch

# print(torch.cuda.is_available())

# pipeline = AutoPipelineForInpainting.from_pretrained(
#     "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
# )
# pipeline.enable_model_cpu_offload()
# # remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed

# #pipeline.enable_xformers_memory_efficient_attention()

# init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
# mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png")

# prompt = "a black cat with glowing eyes, cute, adorable, disney, pixar, highly detailed, 8k"
# negative_prompt = "bad anatomy, deformed, ugly, disfigured"
# image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=init_image, mask_image=mask_image).images[0]
# make_image_grid([init_image, mask_image, image], rows=1, cols=3)


#Code Source:https://huggingface.co/docs/diffusers/using-diffusers/inpaint

#PB : CUDA PAS DISPO
#### DEUXIEME ESSAI #####
import PIL
import requests
import torch
from io import BytesIO

from diffusers import StableDiffusionInpaintPipeline

pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float32,
    use_safetensors=True,
    variant="fp16",
)

# Move the pipeline to the CPU
pipeline = pipeline.to("cpu")

print("Ok?")

#Code Source: https://huggingface.co/docs/diffusers/using-diffusers/inpaint
def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/billow926-12-Wc-Zgx6Y.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/billow926-12-Wc-Zgx6Y_mask.png"

init_image = download_image(img_url).resize((512, 512))
mask_image = download_image(mask_url).resize((512, 512))

print('Ok 2')

prompt = "Two puppies wearing red hats"

new_image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image).images[0]
new_image

new_image.save("output.png")

print("Image saved as output.png")