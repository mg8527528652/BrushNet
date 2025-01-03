from diffusers import BrushNetModel, ControlNetModel, AutoencoderKL, DPMSolverMultistepScheduler
from diffusers.utils import load_image
import numpy as np
import torch
from diffusers.pipelines.brushnet.pipeline_brushnet_controlnet_sdxl import StableDiffusionXLBrushNetControlNetPipeline
import cv2
from PIL import Image, ImageOps
import os
import json
from tqdm import tqdm

def load_control_image(image, mask):
    # Convert image to numpy array if it isn't already
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    # Handle mask processing
    if mask.mode == 'RGBA':
        mask = mask.split()[-1]  # Get alpha channel
    else:
        mask = mask.convert("L")
    mask = np.array(mask)
    mask = mask.astype(np.float32) / 255.0
    
    # Threshold mask
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    
    # Apply mask to image
    masked_image = image * (1 - mask[..., None])  # Broadcasting mask to match image channels
    masked_image = masked_image.astype(np.uint8)
    masked_image = Image.fromarray(masked_image)
    
    # Convert mask to PIL
    mask = Image.fromarray((mask * 255).astype(np.uint8))
    return mask, masked_image


def load_input_images_and_prompt(image_path, prompt_path):
    """Load and prepare input images and mask."""    
    init_image = cv2.imread(image_path)[:,:,::-1]
    mask_image = cv2.imread(image_path, -1)[:,:,3]
    mask_image = (mask_image > 127).astype(np.uint8)
    
    mask_image = 1.*(mask_image > 0)
    # # perform thresholding on mask_image
    mask_image = 1.0 - mask_image
    # control_image, _ = load_control_image(cv2.imread(image_path),  ImageOps.invert(Image.open(image_path).split()[-1]))

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
    
    # add channel to mask image
    # # if prompt path is a json path, load prompt from json. else, use prompt directly
    if os.path.splitext(prompt_path)[-1].lower() == ".json":
        with open(prompt_path, "r") as f:
            prompt = json.load(f)["bg_label"]
    else:
        prompt = prompt_path
    # control_image is the same as mask_image
    # control_image = mask_image
    # save all images
    os.makedirs('init', exist_ok=True)
    os.makedirs('mask', exist_ok=True)
    init_image.save(os.path.join('init', 'file.png'))
    mask_image.save(os.path.join('mask', 'file.png'))
    
    # return init_image, init_image, mask_image, prompt
    return init_image, mask_image, mask_image, prompt   


def load_input_images_and_prompt_5ch(image_path, prompt_path):
    """Load and prepare input images and mask."""    
    init_image = cv2.imread(image_path)[:,:,::-1]
    mask_image = 1.*(cv2.imread(image_path, -1)[:,:,3] > 0)
    mask_image = 1.0 - mask_image

    # resize image
    h,w,_ = init_image.shape
    if w<h:
        scale=1024/w
    else:
        scale=1024/h
    new_h=int(h*scale)
    new_w=int(w*scale)
    # save padding mask
    # save padding mask
    # cv2.imwrite('padding_mask.png', padding_mask) 
    
    init_image=cv2.resize(init_image,(new_w,new_h))
    mask_image=cv2.resize(mask_image,(new_w,new_h))[:,:,np.newaxis]

    init_image = init_image * (1-mask_image)

    with open(prompt_path, "r") as f:
        prompt = json.load(f)["bg_label"]
    # pad image to make it square
    init_image_padded = cv2.copyMakeBorder(init_image, 0, max(0, new_h - h), 0, max(0, new_w - w), cv2.BORDER_CONSTANT, value=(0, 0, 0))
    mask_image_padded = cv2.copyMakeBorder(mask_image, 0, max(0, new_h - h), 0, max(0, new_w - w), cv2.BORDER_CONSTANT, value=(0, 0, 0))
    # create a paddiing mask, with red pixels where padding is added in above step
    padding_mask = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    padding_mask = cv2.copyMakeBorder(padding_mask, 0, max(0, new_h - h), 0, max(0, new_w - w), cv2.BORDER_CONSTANT, value=(255, 0, 0))
    # convert padding mask to 1 channel
    padding_mask = padding_mask[:,:,0]
    padding_mask = padding_mask[:,:,np.newaxis]
    mask_image_padded  = mask_image_padded[:,:,np.newaxis]
    # create a 5 channel tensor with 3channels of init_image_padded, 1 channel of mask_image_padded and 1 channel of padding_mask
    init_image_padded = np.concatenate([init_image_padded, mask_image_padded, padding_mask], axis=-1)
    # convert to tensor
    control_image = torch.from_numpy(init_image_padded).float()
    
    
    init_image = Image.fromarray(init_image.astype(np.uint8)).convert("RGB")
    mask_image = Image.fromarray(mask_image.astype(np.uint8).repeat(3,-1)*255).convert("RGB")
    return init_image, control_image, mask_image, prompt


def initialize_models(brushnet_path, controlnet_path, vae_path="madebyollin/sdxl-vae-fp16-fix"):
    """Initialize BrushNet, ControlNet, and VAE models."""
    brushnet = BrushNetModel.from_pretrained(
        brushnet_path, 
        torch_dtype=torch.float16
    )
    controlnet = ControlNetModel.from_pretrained(
        controlnet_path,
        torch_dtype=torch.float16
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        torch_dtype=torch.float16
    )
    return brushnet, controlnet, vae


def setup_pipeline(base_model_path, brushnet, controlnet, vae, num_inference_steps=50):
    """Set up the BrushNet-ControlNet pipeline."""
    pipe = StableDiffusionXLBrushNetControlNetPipeline.from_pretrained(
        base_model_path,
        brushnet=brushnet,
        controlnet=controlnet,
        vae=vae,
        # num_inference_steps=num_inference_steps,
        torch_dtype=torch.float16
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_lu_lambdas=True,
        use_karras_sigmas=True,
        euler_at_final=True
    )
    # pipe.enable_model_cpu_offload()
    pipe.to("cuda")
    return pipe


def generate_image(pipe, prompt, image, control_image, mask_image, 
                  brushnet_scale=1.0, controlnet_scale=0.5 , 
                  negative_prompt="low quality, bad quality, distorted background, cartoon, cgi, render, 3d, artwork, illustration, 3d render, cinema 4d, artstation, octane render, painting, oil painting, anime, 2d, sketch, drawing, bad photography, bad photo, deviant art"):
    """Generate the final image using the pipeline."""
    gen_image = pipe(
        prompt,
        image=image,
        control_image=control_image,
        mask=mask_image,
        brushnet_conditioning_scale=brushnet_scale,
        controlnet_conditioning_scale=controlnet_scale,
        negative_prompt=negative_prompt,
        num_inference_steps=25
    ).images[0]
    return gen_image


def postprocess_image(image_path, gen_image):
    # load image
    image = cv2.imread(image_path)
    # if gen image is not image, read it
    if isinstance(gen_image, str):
        gen_image = cv2.imread(gen_image, -1)
    # if gen image is PIL image, convert it to opencv image
    if isinstance(gen_image, Image.Image):
        gen_image = np.array(gen_image)
        gen_image = gen_image[:,:,::-1]
    # foreground mask
    mask_image = cv2.imread(image_path, -1)[:,:,3]
    mask_image = (mask_image > 127).astype(np.uint8)
    mask_image = 1.*(mask_image > 0)
    # blur mask image
    # mask_image = cv2.GaussianBlur(mask_image, (21, 21), 0)
    # if mask image lacks 3rd dimension, add it
    if len(mask_image.shape) == 2:
        mask_image = mask_image[:,:,np.newaxis]
    # resize gen image
    h,w,_ = image.shape
    gen_image = cv2.resize(gen_image,(w,h))
    # combine image and gen image
    gen_image = gen_image * (1-mask_image) + image * mask_image
    return gen_image


def main():
    # Configuration
    alpha_images_path = "/root/BrushNet/BENCHMARK_DATASET/alpha_images"
    brushnet_path = "/root/BrushNet/checkpoints/brushnet/segmentation_mask_brushnet_ckpt_sdxl_v1"
    controlnet_path = "/root/BrushNet/checkpoints/cn_train_inpaint_sdxl_v2/checkpoint-89000/controlnet"
    base_model_path = "SG161222/RealVisXL_V4.0"
    # base_model_path = "RunDiffusion/Juggernaut-XL"
    prompt_json_path = "/root/BrushNet/BENCHMARK_DATASET/bg_prompts"
    output_path = "results/output_images-sdxl-cn-sdxlinpaint-base/"
    brushnet_scale = 1.0
    controlnet_scale = 0.8
    
    os.makedirs(output_path, exist_ok=True)
        # Initialize models
    brushnet, controlnet, vae = initialize_models(brushnet_path, controlnet_path)
            # Setup pipeline
    pipe = setup_pipeline(base_model_path, brushnet, controlnet, vae)

    for cn_scale in [0.00, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        output_path_cn = os.path.join(output_path, f"cn_scale_{cn_scale}")
        os.makedirs(output_path_cn, exist_ok=True)
        controlnet_scale = cn_scale
        for file in tqdm(os.listdir(alpha_images_path)):
            try:
                image_path = os.path.join(alpha_images_path, file)
                prompt_path = os.path.join(prompt_json_path, file.replace(".png", ".json"))
                # Load images
                image, control_image, mask_image, prompt = load_input_images_and_prompt(image_path, prompt_path)
                # image, control_image, mask_image, prompt = load_input_images_and_prompt_5ch(image_path, prompt_path)

                # Generate and save image
                output_image = generate_image(pipe, prompt, image, control_image, mask_image, brushnet_scale, controlnet_scale)
                # postprocess image
                output_image = postprocess_image(image_path, output_image)
                # output_image.save(os.path.join(output_path, file))
                cv2.imwrite(os.path.join(output_path_cn, file), output_image)
            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue


def main_test():
    images_path = "/root/BrushNet/BENCHMARK_DATASET/alpha_images"
    gen_images_path = "output_images-rv-0.5"
    for file in tqdm(os.listdir(images_path)):
        try:
            image_path = os.path.join(images_path, file)
            gen_image_path = os.path.join(gen_images_path, file)
            output_image = postprocess_image(image_path, gen_image_path)
            # output_image.save(os.path.join(gen_images_path, file))
            os.makedirs('blend', exist_ok=True)
            cv2.imwrite(os.path.join('blend', file), output_image)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
if __name__ == "__main__":
    main()
    # main_test()

