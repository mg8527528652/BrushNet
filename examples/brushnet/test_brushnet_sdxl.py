import os
from diffusers import StableDiffusionXLBrushNetPipeline, BrushNetModel, DPMSolverMultistepScheduler, UniPCMultistepScheduler, AutoencoderKL
import torch
import cv2
import numpy as np
from PIL import Image
import json
# Input source folders paths
images_folder = "/home/ubuntu/mayank/benchmark_dataset_BG_REPLACOR/images_sorted/"
masks_folder = "/home/ubuntu/mayank/benchmark_dataset_BG_REPLACOR/masks/"
alpha_folder = "/home/ubuntu/mayank/benchmark_dataset_BG_REPLACOR/alpha_images/"
prompt_folder = "/home/ubuntu/mayank/benchmark_dataset_BG_REPLACOR/bg_prompts/"
output_folder = "output_images-rv/"  # Where generated images will be saved

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Rest of your model initialization code remains the same
base_model_path = "RunDiffusion/Juggernaut-XL"
brushnet_path = "/home/ubuntu/mayank/BrushNet/segmentation_mask_brushnet_ckpt_sdxl_v1"
blended = False
brushnet_conditioning_scale = 1.0

# Initialize models (same as before)
brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch.float16)
pipe = StableDiffusionXLBrushNetPipeline.from_pretrained(
    base_model_path, brushnet=brushnet,
    torch_dtype=torch.float16, 
    # use_safetensors=True
)
pipe.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,
                                                          **{"use_lu_lambdas": True,
                                                              "use_karras_sigmas": True,
                                                              "euler_at_final": True})
# pipe.enable_model_cpu_offload()
pipe.to('cuda')

def process_image(image_filename, prompt):
    # Construct full paths
    image_path = os.path.join(images_folder, image_filename)
    mask_path = os.path.join(masks_folder, image_filename)
    output_path = os.path.join(output_folder, image_filename)
    
    # Your caption - you might want to customize this per image
    caption = prompt
    
    init_image = cv2.imread(image_path)[:,:,::-1]
    mask_image = 1.*(cv2.imread(mask_path).sum(-1)>255)
    mask_image = 1.0 - mask_image

    # resize image
    h,w,_ = init_image.shape
    if w<h:
        scale=1024/w
    else:
        scale=1024/h
    new_h=int(h*scale)
    new_w=int(w*scale)

    init_image=cv2.resize(init_image,(new_w,new_h))
    mask_image=cv2.resize(mask_image,(new_w,new_h))[:,:,np.newaxis]

    init_image = init_image * (1-mask_image)

    init_image = Image.fromarray(init_image.astype(np.uint8)).convert("RGB")
    mask_image = Image.fromarray(mask_image.astype(np.uint8).repeat(3,-1)*255).convert("RGB")

    generator = torch.Generator("cuda").manual_seed(4321)

    image = pipe(
        prompt=caption, 
        image=init_image, 
        mask=mask_image, 
        num_inference_steps=25, 
        generator=generator,
        brushnet_conditioning_scale=brushnet_conditioning_scale
    ).images[0]

    if blended:
        image_np=np.array(image)
        init_image_np=cv2.imread(image_path)[:,:,::-1]
        mask_np = 1.*(cv2.imread(mask_path).sum(-1)>255)[:,:,np.newaxis]
        mask_image = 1.0 - mask_image

        # Resize init_image_np and mask_np to match image_np dimensions
        image_h, image_w = image_np.shape[:2]
        init_image_np = cv2.resize(init_image_np, (image_w, image_h))
        mask_np = cv2.resize(mask_np, (image_w, image_h), interpolation=cv2.INTER_NEAREST)
        mask_np = mask_np[:,:,np.newaxis]

        mask_blurred = cv2.GaussianBlur(mask_np*255, (21, 21), 0)/255
        mask_blurred = mask_blurred[:,:,np.newaxis]
        mask_np = 1-(1-mask_np) * (1-mask_blurred)

        image_pasted=init_image_np * (1-mask_np) + image_np*mask_np
        image_pasted=image_pasted.astype(image_np.dtype)
        image=Image.fromarray(image_pasted)

    image.save(output_path)

# Process all images in the folder
for image_file in os.listdir(images_folder):
    if image_file.endswith(('.jpg', '.jpeg', '.png')):
        print(f"Processing {image_file}...")
        try:
            prompt_file = image_file.replace(".jpg", ".json")
            prompt_path = os.path.join(prompt_folder, prompt_file)
            prompt = json.load(open(prompt_path))["bg_label"]
            process_image(image_file, prompt)
            print(f"Successfully processed {image_file}")
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
