import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
                        
from . import utils

import os
import tempfile
import datetime
from tqdm.auto import tqdm
import csv

import numpy as np
import math
import random
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageOps
import cv2

import torch
import torchvision
from torchvision import transforms
from diffusers import AutoPipelineForInpainting
from diffusers.models.attention_processor import AttnProcessor2_0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



import csv

def replace_vehicle_segment(canvas_image, outpainted_image, model, config, detection_threshold=0.7):
    """
    Detects vehicles in the canvas_image using Mask R-CNN and replaces the corresponding
    pixels in outpainted_image with pixels from canvas_image.
    
    Vehicles include car, motorcycle, bus, and truck (COCO category IDs: 3, 4, 6, 8).
    Detections with confidence below detection_threshold are ignored.
    
    If no vehicle is detected, returns outpainted_image unchanged.
    
    Parameters:
      canvas_image (PIL.Image): The canvas image.
      outpainted_image (PIL.Image): The outpainted image.
      device (str, optional): Device to run the model on. If None, uses "cuda" if available.
      detection_threshold (float): Minimum confidence for detections (default 0.7).
    
    Returns:
      PIL.Image: Modified outpainted image.
    """

    
    # Transform: convert PIL image to tensor.
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    input_tensor = transform(canvas_image).to(device)
    
    # Run inference.
    with torch.no_grad():
        predictions = model([input_tensor])[0]
    
    # COCO category IDs for vehicles (using standard mapping):
    # 3: car, 6: bus, 8: truck.
    vehicle_labels = {3, 6, 8}
    
    # Initialize an empty mask.
    if predictions['masks'].shape[0] == 0:
        return outpainted_image

    

    combined_mask = torch.zeros_like(predictions['masks'][0][0], dtype=torch.uint8)
    # Loop over predictions and combine vehicle masks.
    for mask, label, score in zip(predictions['masks'], predictions['labels'], predictions['scores']):
        if score < detection_threshold:
            continue
        if label.item() in vehicle_labels:
            # Each mask is of shape [1, H, W]. Threshold it (0.5 is common).
            binary_mask = (mask[0] > 0.5).to(torch.uint8)
            combined_mask |= binary_mask  # Combine masks with bitwise OR.
    
    # If no vehicles were detected, return outpainted image.
    if combined_mask.sum() == 0:
        return outpainted_image

    # Convert images to numpy arrays.
    canvas_np = np.array(canvas_image)
    outpaint_np = np.array(outpainted_image)
    # Convert combined_mask to float32 (0.0 or 1.0) and use outpainted image as guidance.
    combined_mask_float = combined_mask.detach().cpu().numpy().astype(np.float32)
    outpaint_np_float = outpaint_np.astype(np.float32)
    canvas_np_float = canvas_np.astype(np.float32)
    
    # Apply guided filter (using a radius of 5 and a small eps; adjust as needed)
    refined_mask = cv2.ximgproc.guidedFilter(guide=canvas_np_float,
                                             src=combined_mask_float,
                                             radius=config['segmentation']['guided_filter']['radius'],
                                             eps=config['segmentation']['guided_filter']['eps'])

    # Threshold the refined mask to get a binary mask
    combined_mask = refined_mask > config['segmentation']['guided_filter']['threshold']
    #display(Image.fromarray((combined_mask.astype(np.uint8) * 255)))
    # Convert the combined mask to a numpy array.
    vehicle_mask_np = combined_mask
    
    # Convert images to numpy arrays.
    canvas_np = np.array(canvas_image)
    outpaint_np = np.array(outpainted_image)
    
    # Replace pixels in the outpainted image with canvas pixels where the mask is True.
    vehicle_segment = canvas_np[vehicle_mask_np == 1]
    if config['outpainting']['recolor_object']:
        # Ramdomly permute channels
        #vehicle_segment = vehicle_segment[..., np.random.permutation(3)]
        vehicle_segment = vehicle_segment[..., random.choice([[0,2,1], [1,0,2], [1,2,0], [2,0,1], [2,1,0]])]


    outpaint_np[vehicle_mask_np == 1] = vehicle_segment
    return Image.fromarray(outpaint_np)


def generate_mask_canvas(seed_image, class_name, config):
    # Access the class_index dictionary
    class_index = config['class_index']
    
    canvas_size=(config['outpainting']['canvas']['size']['width'],config['outpainting']['canvas']['size']['height'])

    image = Image.open(seed_image).convert("RGB")
    #image = Image.fromarray(np.random.permutation(np.array(image).transpose(2, 0, 1)).transpose(1, 2, 0))
    # Randomly scale the image, create a new image with transparency, etc.
    canvas_area = canvas_size[0] * canvas_size[1]
    area_ratio = random.uniform(config['outpainting']['canvas']['area_ratio']['min'], config['outpainting']['canvas']['area_ratio']['max'])
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
    width_norm /= (1 + config['detection']['buffer'])
    height_norm /= (1 + config['detection']['buffer'])
    
    annotation = f"{class_index[class_name]} {x_center_norm} {y_center_norm} {width_norm} {height_norm}\n"
    
    return canvas_image, mask, annotation, area_ratio


def generate_outpainted(seed_folder, output_folder, config, prompt = None, save_interim = False, random_seed = None):
    
    os.makedirs(output_folder, exist_ok=True)
    csv_file = os.path.join(output_folder, 'outpainted_prompts.csv')
    
    # Write header only if file doesn't exist or is empty
    if not os.path.exists(csv_file) or os.stat(csv_file).st_size == 0:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['split_set', 'image_name', 'outpainted_file_name', 'prompt', 'qualiclip_score', 'clip_iqa_score', 'brisque_score', 'blur_factor_coef', 'iteration', 'guidance_scale', 'area_ratio'])

    # Prepare failures CSV
    failures_csv = os.path.join(output_folder, 'outpainting_failures.csv')
    if not os.path.exists(failures_csv) or os.stat(failures_csv).st_size == 0:
        with open(failures_csv, 'w', newline='') as f_fail:
            writer = csv.writer(f_fail)
            writer.writerow(['split_set', 'image_name'])
            
    # Load pretrained Mask R-CNN model.
    segmentation_model_ins = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    segmentation_model_ins.to(device)
    segmentation_model_ins.eval()
    
    # Instantiate Image Assessment Class
    ia = utils.ImageAssessment(device=device)

    pipeline = AutoPipelineForInpainting.from_pretrained(
        config['outpainting']['model'],
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        safety_checker = None,
        requires_safety_checker = False,
    )
    
    #runwayml/stable-diffusion-inpainting
    pipeline.to(device)
    pipeline.unet.set_attn_processor(AttnProcessor2_0())
    pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)
    pipeline.set_progress_bar_config(disable=True)
    
    split_sets = ['train', 'val', 'test']
    #split_sets = ['train']


    
    #select = ['bus_20250408184909984','truck_20250408184934101','coupe_20250408184942117','pickup_20250408184926031','minibus_20250408184918228']
    #select = ['coupe_20250510104424450.jpg', 'bus_20250510104428791.jpg', 'truck_20250510104420183.jpg', 'pickup_20250510104433025.jpg']
    for split_set in split_sets:
        image_folder = os.path.join(seed_folder, split_set, 'images')
        if os.path.exists(image_folder):
            for image_name in tqdm(sorted(os.listdir(image_folder)), position=0, leave=True):
                #if all(sub not in image_name for sub in select): continue
                prompt_location_list = config['outpainting']['prompt_location_list'] 
                prompt_time_list = config['outpainting']['prompt_time_list']
                negative_prompt = config['outpainting']['negative_prompt']
                
                if random_seed is not None: utils.set_seeds(random_seed)
                
                # skip if already succeeded or already failed
                if ((os.path.exists(csv_file) and any(row[1] == image_name
                     for row in csv.reader(open(csv_file,  'r', newline=''))))
                 or (os.path.exists(failures_csv) and any(row[1] == image_name
                     for row in csv.reader(open(failures_csv, 'r', newline=''))))):
                    continue
                
                class_name =image_name.split("_")[0]

                if class_name=='background': continue
                if class_name=="bus": prompt_location_list.extend(['downtown'])
                if class_name=='truck': prompt_location_list.extend(['highway'])
                
                
                image_path = os.path.join(image_folder, image_name)
                timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')[:17]
                outpainted_file_name = f"{os.path.splitext(image_name)[0]}_{timestamp}.png"
                output_images_dir = os.path.join(output_folder, split_set, "images")
                output_labels_dir = os.path.join(output_folder, split_set, "labels")
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

                seed_image = Image.open(image_path).convert("RGB")
                
                qualiclip_score_threshold = config['outpainting']['quality']['qualiclip']['upper_min']
                qualiclip_ok = False
                
                outpainted_image_list = []
                for iteration in range(1, config['outpainting']['max_iter']):
                    canvas_image, mask_image, annotation, area_ratio = generate_mask_canvas(image_path,class_name,config)
                    blur_factor_coef = random.randint(config['outpainting']['mask_blur_coef']['min'],config['outpainting']['mask_blur_coef']['max'])
                    blur_factor = blur_factor_coef*np.log(2*area_ratio+1)
                    mask_image_blurred = pipeline.mask_processor.blur(mask_image, blur_factor=blur_factor)
                    
                    if not prompt:
                        prompt_location = random.choice(prompt_location_list)
                        prompt_time = random.choice(prompt_time_list)
                        prompt = "A photo of a " + prompt_location + " during " + prompt_time
                        
                    guidance_scale = random.randint(config['outpainting']['guidance_scale']['min'],config['outpainting']['guidance_scale']['max'])
                    outpainted_image = pipeline(prompt=prompt + " with no vehicle, detailed.",
                                                negative_prompt = negative_prompt,
                                                image=canvas_image,
                                                mask_image=mask_image_blurred,
                                                height=canvas_image.height,
                                                width=canvas_image.width,
                                                strength=1,
                                                guidance_scale = guidance_scale,
                                        ).images[0]
                    outpainted_image= replace_vehicle_segment(canvas_image, outpainted_image, segmentation_model_ins, config=config)
                    
                    brisque_score = ia.calculate_brisque(outpainted_image)
                    clip_iqa_score = ia.calculate_clip_iqa(outpainted_image)
                    if brisque_score<config['outpainting']['quality']['brisque']['max'] and brisque_score>0 and \
                        clip_iqa_score>config['outpainting']['quality']['clip_iqa']['min']:
                        qualiclip_score = ia.calculate_qualiclip(outpainted_image)
                        outpainted_image_list.append((outpainted_image,annotation,prompt,qualiclip_score,blur_factor_coef,guidance_scale,area_ratio))
                        if qualiclip_score > qualiclip_score_threshold: qualiclip_ok = True
                        if (qualiclip_ok and iteration>=config['outpainting']['min_iter']) or (outpainted_image_list and iteration>=config['outpainting']['mid_iter']):
                            break
                        else:
                            qualiclip_score_threshold = max(qualiclip_score_threshold - config['outpainting']['quality']['qualiclip']['step'],
                                                            config['outpainting']['quality']['qualiclip']['lower_min'])
                            
                    prompt = None

                    
                # pick the outpainted image with the highest qualiclip score, or None if the list is empty
                best = max(
                    outpainted_image_list,
                    key=lambda x: x[3],
                    default=None
                )
                # if we have a candidate and its score clears the threshold
                if best: # and best[3] > qualiclip_score_threshold:
                    outpainted_image, annotation, prompt, qualiclip_score, blur_factor_coef, guidance_scale, area_ratio = best
                    outpainted_image.save(outpainted_file_path)
                    with open(annotation_path, 'w') as file:
                        file.write(annotation)
                    if save_interim:
                        canvas_image.save(canvas_file_path)
                        mask_image_blurred.save(mask_file_path)
                    clip_iqa_score = ia.calculate_clip_iqa(outpainted_image)
                    brisque_score = ia.calculate_brisque(outpainted_image)
                    with open(csv_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([split_set, image_name, outpainted_file_name, prompt, qualiclip_score, clip_iqa_score, brisque_score, blur_factor_coef, iteration, guidance_scale, area_ratio])
                # otherwise log a failure
                else:
                    # Log failures
                    with open(failures_csv, 'a', newline='') as f_fail:
                        writer = csv.writer(f_fail)
                        writer.writerow([split_set, image_name])
                        
                prompt = None
                '''
                display(outpainted_image)
                brisque_score = ia.calculate_brisque(outpainted_image)
                clip_iqa_score = ia.calculate_clip_iqa(outpainted_image)
                print(prompt, blur_factor_coef)
                print('Succuessful: clip:', clip_iqa_score, 'brisque:', brisque_score, 'qualiclip', qualiclip_score)
                '''

