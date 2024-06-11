import os
import random
from PIL import Image, ImageDraw, ImageOps
import datetime

import torch
import numpy as np
import PIL
from diffusers import AutoPipelineForInpainting

from diffusers.utils import load_image, make_image_grid
import glob
from tqdm import tqdm
import numpy as np

from . import utils

def generate_mask_canvas(seed_image, class_name, canvas_size=(512, 512)):
    class_index = {
        'passenger_car': 0,
        'utility_vehicle': 1,
        'bus': 2,
        'van': 3,
        'pickup_truck': 4,
        'heavy_truck': 5,
    }    
    image = Image.open(seed_image).convert("RGB")

    # Randomly scale the image, create a new image with transparency, etc.
    canvas_area = canvas_size[0] * canvas_size[1]
    area_ratio = random.uniform(0.1, 0.2)
    scale_factor = (area_ratio * canvas_area) / (image.width * image.height)
    scale_factor = scale_factor ** 0.5
    new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
    image = image.resize(new_size, Image.Resampling.LANCZOS)

    canvas_image = Image.new('RGB', canvas_size, (255, 255, 255))
    max_x_offset = canvas_size[0] - image.width
    max_y_offset = canvas_size[1] - image.height
    offset_x = random.randint(0, max(max_x_offset, 0))
    offset_y = random.randint(int(canvas_size[1] * 1/3), max(max_y_offset, int(canvas_size[1] * 1/3)))
    canvas_image.paste(image, (offset_x, offset_y))

    mask = Image.new('RGB', canvas_size, (255, 255, 255))
    draw = ImageDraw.Draw(mask)
    draw.rectangle([offset_x, offset_y, offset_x + image.width, offset_y + image.height], fill=(0, 0, 0))

    # Mirror the canvas randomly to increase variety
    if random.choice([True, False]):
        canvas_image = ImageOps.mirror(canvas_image)
        mask = ImageOps.mirror(mask)
        offset_x = canvas_size[0] - offset_x - image.width

    # Save the annotation
    # Calculate normalized values
    x_center_norm = (offset_x + image.width / 2) / canvas_size[0]
    y_center_norm = (offset_y + image.height / 2) / canvas_size[1]
    width_norm = image.width / canvas_size[0]
    height_norm = image.height / canvas_size[1]
    # Scale down the bounding box dimensions by reversing 1.10 scaling
    width_norm /= 1.1
    height_norm /= 1.1
    
    annotation = f"{class_index[class_name]} {x_center_norm} {y_center_norm} {width_norm} {height_norm}\n"
    
    return canvas_image, mask, annotation, area_ratio


def generate_outpainted(seed_folder,
                        output_folder,
                        max_iter = 5,
                        prompt = None,
                        save_interim = False):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Instantiate Image Assessment Class
    ia = utils.ImageAssessment(device=device)

    pipeline = AutoPipelineForInpainting.from_pretrained(
        "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, variant="fp16"
    )
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    categories = ['train', 'val', 'test']

    for category in categories:
        #if category!='test': continue
        category_path = os.path.join(seed_folder, category)
        if os.path.exists(category_path):
            prompt_time_list = ['spring', 'summer', 'fall', 'winter',
                                'a sunny day', 'a cloudy day', 'a rainy day',
                                'evening', 'sunset', 'sunrise']
            for class_name in tqdm(sorted(os.listdir(category_path))):
                #if class_name not in ['utility_vehicle', 'van']: continue
                
                if class_name=="heavy_truck":
                    prompt_location_list = ['highway', 'road', 'street']
                elif class_name=="bus":
                    prompt_location_list = ['street', 'downtown', 'plaza']
                else:
                    prompt_location_list = ['road','street','downtown']
                
                
                class_path = os.path.join(category_path, class_name)
                for image_name in sorted(os.listdir(class_path)):
                    image_path = os.path.join(class_path, image_name)
                    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')[:17]
                    outpainted_file_name = f"{os.path.splitext(image_name)[0]}_{timestamp}.png"
                    output_images_dir = os.path.join(output_folder, category, "images")
                    output_labels_dir = os.path.join(output_folder, category, "labels")
                    os.makedirs(output_images_dir, exist_ok=True)
                    os.makedirs(output_labels_dir, exist_ok=True)
                    outpainted_file_path = os.path.join(output_images_dir, outpainted_file_name)
                    annotation_path = os.path.join(output_labels_dir, outpainted_file_name.replace('.png', '.txt'))
                    
                    if save_interim:
                        canvas_images_dir = output_images_dir.replace('outpainted', 'canvas')
                        mask_images_dir = output_images_dir.replace('outpainted', 'mask')
                        os.makedirs(canvas_images_dir, exist_ok=True)
                        os.makedirs(mask_images_dir, exist_ok=True)
                        canvas_file_name = f"{os.path.splitext(image_name)[0]}_{timestamp}_canvas.png"
                        mask_file_name = f"{os.path.splitext(image_name)[0]}_{timestamp}_mask.png"
                        canvas_file_path = os.path.join(canvas_images_dir, canvas_file_name)
                        mask_file_path = os.path.join(mask_images_dir, mask_file_name)

                    
                    attempt = 0
                    
                    clip_iqa_score_threshold = 0.95
                    brisque_score_threshold = 10
                    
                    while attempt < max_iter:
                        canvas_image, mask_image, annotation, area_ratio = generate_mask_canvas(image_path,class_name)
                        blur_factor_coef = random.randint(100,130)
                        blur_factor = blur_factor_coef*area_ratio**0.5
                        mask_image_blurred = pipeline.mask_processor.blur(mask_image, blur_factor=blur_factor)
                        
                        if not prompt:
                            prompt_location = random.choice(prompt_location_list)
                            prompt_time = random.choice(prompt_time_list)
                            prompt = "A " + prompt_location + " during " + prompt_time + " with no vehicle."
                            
                        negative_prompt = "billboard, advertisement, text, traffic, train, car, truck, bus, van."
                        
                        outpainted_image = pipeline(prompt=prompt,
                                                 negative_prompt = negative_prompt,
                                                 image=canvas_image,
                                                 mask_image=mask_image_blurred,
                                                 height=canvas_image.height,
                                                 width=canvas_image.width,
                                            ).images[0]
                        clip_iqa_score = ia.calculate_clip_iqa(outpainted_image)
                        if clip_iqa_score >= clip_iqa_score_threshold:
                            similarity_score = ia.calculate_ssim(canvas_image, outpainted_image, mask_image)
                            if similarity_score >= 0.93:
                                brisque_score = ia.calculate_brisque(outpainted_image)
                                if brisque_score <= brisque_score_threshold:
                                    outpainted_image.save(outpainted_file_path)
                                    with open(annotation_path, 'w') as file:
                                        file.write(annotation)
                                    if save_interim:
                                        canvas_image.save(canvas_file_path)
                                        mask_image_blurred.save(mask_file_path)
                                    prompt = None
                                    break
                        attempt += 1
                        clip_iqa_score_threshold = max(0.9,clip_iqa_score_threshold-0.0025)
                        brisque_score_threshold = min(14.9,brisque_score_threshold+0.5)