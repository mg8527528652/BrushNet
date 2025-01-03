import os
from diffusers import StableDiffusionXLBrushNetPipeline, BrushNetModel, DPMSolverMultistepScheduler, AutoencoderKL
import torch
import cv2
import numpy as np
from PIL import Image
import json
from tqdm import tqdm

def load_input_images_and_prompt(image_path, mask_path, prompt):
    """Load and prepare input images and mask."""
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
    
    return init_image, mask_image

def initialize_models_brushnet(base_model_path, brushnet_path):
    """Initialize BrushNet and pipeline models."""
    brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch.float16)
    pipe = StableDiffusionXLBrushNetPipeline.from_pretrained(
        base_model_path, 
        brushnet=brushnet,
        torch_dtype=torch.float16
    )
    pipe.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_lu_lambdas=True,
        use_karras_sigmas=True,
        euler_at_final=True
    )
    pipe.to('cuda')
    return pipe

def generate_image_brushnet(pipe, prompt, init_image, mask_image, brushnet_conditioning_scale=1.0):
    """Generate the final image using the pipeline."""
    generator = torch.Generator("cuda").manual_seed(4321)
    
    return pipe(
        prompt=prompt, 
        image=init_image, 
        mask=mask_image, 
        num_inference_steps=25, 
        generator=generator,
        brushnet_conditioning_scale=brushnet_conditioning_scale
    ).images[0]

def postprocess_image_brushnet(image, init_image_path, mask_path, blended=False):
    """Post-process the generated image with optional blending."""
    if not blended:
        return image
        
    image_np = np.array(image)
    init_image_np = cv2.imread(init_image_path)[:,:,::-1]
    mask_np = 1.*(cv2.imread(mask_path).sum(-1)>255)[:,:,np.newaxis]
    mask_image = 1.0 - mask_np

    # Resize init_image_np and mask_np to match image_np dimensions
    image_h, image_w = image_np.shape[:2]
    init_image_np = cv2.resize(init_image_np, (image_w, image_h))
    mask_np = cv2.resize(mask_np, (image_w, image_h), interpolation=cv2.INTER_NEAREST)
    mask_np = mask_np[:,:,np.newaxis]

    mask_blurred = cv2.GaussianBlur(mask_np*255, (21, 21), 0)/255
    mask_blurred = mask_blurred[:,:,np.newaxis]
    mask_np = 1-(1-mask_np) * (1-mask_blurred)

    image_pasted = init_image_np * (1-mask_np) + image_np*mask_np
    return Image.fromarray(image_pasted.astype(image_np.dtype))

def main():
    # Configuration
    images_folder = "/root/BrushNet/BENCHMARK_DATASET/images_sorted"
    masks_folder = "/root/BrushNet/BENCHMARK_DATASET/masks"
    prompt_folder = "/root/BrushNet/BENCHMARK_DATASET/bg_prompts"
    output_folder = "output_images-rv2/"
    base_model_path = "RunDiffusion/Juggernaut-XL"
    brushnet_path = "/root/BrushNet/checkpoints/brushnet/segmentation_mask_brushnet_ckpt_sdxl_v1"
    blended = False
    brushnet_conditioning_scale = 1.0

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Initialize models
    pipe = initialize_models_brushnet(base_model_path, brushnet_path)

    # Process all images
    for image_file in tqdm(os.listdir(images_folder)):
        if not image_file.endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        try:
            # Load prompt
            prompt_file = image_file.replace(".jpg", ".json")
            prompt_path = os.path.join(prompt_folder, prompt_file)
            prompt = json.load(open(prompt_path))["bg_label"]

            # Prepare paths
            image_path = os.path.join(images_folder, image_file)
            mask_path = os.path.join(masks_folder, image_file)
            output_path = os.path.join(output_folder, image_file)

            # Load and prepare images
            init_image, mask_image = load_input_images_and_prompt(image_path, mask_path, prompt)

            # Generate image
            generated_image = generate_image_brushnet(pipe, prompt, init_image, mask_image, brushnet_conditioning_scale)

            # Post-process and save
            final_image = postprocess_image_brushnet(generated_image, image_path, mask_path, blended)
            final_image.save(output_path)

            print(f"Successfully processed {image_file}")
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")

if __name__ == "__main__":
    main()
