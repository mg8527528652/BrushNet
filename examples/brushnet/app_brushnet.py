import gradio as gr
import numpy as np
import cv2
import os
from test_brushnet_sdxl import *
from prompt_enhancer import PromptEnhancer
from PIL import Image
import torch

def setup_pipelines_bcn():
    brushnet_path = "/root/BrushNet/checkpoints/brushnet/segmentation_mask_brushnet_ckpt_sdxl_v1"   
    # base_model_path = "SG161222/RealVisXL_V4.0"
    # pipe_bcn_realvis_sdxl_inpaint_cn = initialize_models_brushnet(base_model_path, brushnet_path)
    base_model_path = "RunDiffusion/Juggernaut-XL"
    pipe_bcn_jugger_sdxl_inpaint_cn = initialize_models_brushnet(base_model_path, brushnet_path)
    base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe_bcn_sdxl_sdxl_inpaint_cn = initialize_models_brushnet(base_model_path, brushnet_path)
    # return pipe_bcn_realvis_sdxl_inpaint_cn, pipe_bcn_sdxl_sdxl_inpaint_cn
    return pipe_bcn_sdxl_sdxl_inpaint_cn, pipe_bcn_jugger_sdxl_inpaint_cn

 
def overlay_image(image, gen_image, mask):
    # blend image and gen_image using mask. keep foreground of image and background of gen_image
    # image, gen_image, and mask are PIL images
    gen_image = gen_image.resize(image.size)
    image = image.convert("RGBA")
    gen_image = gen_image.convert("RGBA")
    
    # Invert the mask since we want to keep background from gen_image
    mask = mask.convert("L")  # Convert mask to grayscale
    # mask = Image.eval(mask, lambda x: 255 - x)  # Invert the mask values
    
    # Create an alpha composite image using the inverted mask
    blended_image = Image.composite(image, gen_image, mask)
    return blended_image
    
    
def process_image(image, prompt, brushnet_conditioning_scale):
    try:
        os.makedirs("temp", exist_ok=True)
        brushnet_conditioning_scale = float(brushnet_conditioning_scale)
        # Save input image
        temp_input_path = os.path.join("temp", "input.png")
        if len(image.shape) == 3 and image.shape[-1] == 4:
            cv2.imwrite(temp_input_path, cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA))
        else:
            cv2.imwrite(temp_input_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        img = Image.open(temp_input_path)
        
        # enhance prompt
        api_key = 's'
        prompt_enhancer = PromptEnhancer(api_key=api_key)
        prompt = prompt_enhancer.enhance_prompt(prompt)
        print(prompt)
        # Generate bcn realvis result
        temp_bcn_realvis_path = os.path.join("temp", f"output_bcn_realvis.png")
        init_image, mask_image = load_input_images_and_prompt(temp_input_path)
        bcn_realvis_res = generate_image_brushnet(pipe=pipe_bcn_realvis_sdxl_inpaint_cn, prompt=prompt, init_image=init_image, mask_image=mask_image, brushnet_conditioning_scale=brushnet_conditioning_scale )
        bcn_realvis_res = postprocess_image_brushnet(bcn_realvis_res, temp_input_path)
        # bcn_realvis_res = overlay_image(img.convert("RGB"), bcn_realvis_ res, img.split()[-1])
        # save bcn_realvis_res
        # cv2.imwrite(temp_bcn_realvis_path, bcn_realvis_res)

        
        # generate jugger result
        temp_bcn_jugger_path = os.path.join("temp", f"output_bcn_jugger.png")
        bcn_jugger_res = generate_image_brushnet(pipe=pipe_bcn_jugger_sdxl_inpaint_cn, prompt=prompt, init_image=init_image, mask_image=mask_image, brushnet_conditioning_scale=brushnet_conditioning_scale )
        bcn_jugger_res = postprocess_image_brushnet(bcn_jugger_res, temp_input_path)
        # bcn_jugger_res = overlay_image(img.convert("RGB"), bcn_jugger_res, img.split()[-1])
        # save bcn_jugger_res
        # cv2.imwrite(temp_bcn_jugger_path, bcn_jugger_res)
        
        # Normalize the image array to uint8 range (0-255)
        # bcn_jugger_res = (bcn_jugger_res * 255).clip(0, 255).astype(np.uint8)

        # Clean up temporary files
        # for path in [temp_input_path, temp_bcn_realvis_path, temp_bcn_jugger_path]:
        #     if os.path.exists(path):
        #         try:
        #             os.remove(path)
        #         except:
        #             pass
        # bcn_realvis_res = Image.open(temp_bcn_realvis_path)
        # bcn_jugger_res = Image.open(temp_bcn_jugger_path)
        return [bcn_realvis_res, bcn_jugger_res]
    
    except Exception as e:
        print(f"Error in process_image: {str(e)}")
        raise gr.Error("An error occurred while processing the image. Please try again.")

app = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="numpy", label="Input Image", image_mode='RGBA'),  # Specify RGBA mode
        gr.Textbox(label="Prompt"),
        gr.Slider(minimum=0, maximum=1, step=0.01, value=1.0, label="ControlNet Scale"),
    ],
    outputs=gr.Gallery(label="Output Images"),
    title="Image Processing App",
    description="Upload an image and provide a prompt to process the image.",
    # queue=True,  # Add this line
    max_batch_size=1,  # Add this line
    
    # Increase timeout settings
    # queue_timeout=300,  # 5 minutes queue timeout
    # server_kwargs={
    #     "timeout": 300  # 5 minutes server timeout
    # }
)


if __name__ == "__main__":
    pipe_bcn_realvis_sdxl_inpaint_cn, pipe_bcn_jugger_sdxl_inpaint_cn = setup_pipelines_bcn()
    app.launch(share=True,  max_threads=1,server_port=7867)