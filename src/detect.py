import os
import torch
from torchvision import transforms
import torchvision.models.detection as detection
from PIL import Image
import datetime
import inspect
import numpy as np
from tqdm import tqdm

import yaml

def convert_bbox_coco2yolo(img_width, img_height, bbox):
    # Convert bounding box from COCO format to YOLO format
    x_tl, y_tl, w, h = bbox
    dw = 1.0 / img_width
    dh = 1.0 / img_height
    x_center = x_tl + w / 2.0
    y_center = y_tl + h / 2.0
    x = x_center * dw
    y = y_center * dh
    w = w * dw
    h = h * dh
    return [x, y, w, h]

def detect_vehicles_and_crop(image_path, models, config, device):
    class_index = config['class_index']
    # Detect vehicles in an image using the provided model
    image = Image.open(image_path)
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)

    max_area = 0
    largest_box = None
    iterator = iter(models)
    while largest_box is None:
        key = next(iterator, None)
        if key is None: break
        model = models[key]
        model.to(device).eval()
        with torch.no_grad():
            preds = model(image_tensor)

        # Filter out predictions for vehicles
        vehicle_classes = [3, 6, 8]  # COCO classes for vehicles
        pred = preds[0]  # Get predictions for the first (and only) image in the batch
        #print("Label indices from model predictions:", np.unique(pred['labels'].cpu().numpy()))
        labels = pred['labels'].cpu().numpy()
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()

        # Find the largest box

        for label, box, score in zip(labels, boxes, scores):
            if label in vehicle_classes and score >= 0.33:  # threshold can be adjusted
                x1, y1, x2, y2 = box
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    largest_box = box

    # Crop and return the image within the largest box if it meets the dimension criteria
    if largest_box is not None:
        x1, y1, x2, y2 = largest_box
        # Calculate the center and dimensions of the original box
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1

        # Define scale factor (e.g., 1.1 for 10% increase)
        scale_factor = 1 + config['detection']['buffer']

        # Calculate new width and height
        new_width = width * scale_factor
        new_height = height * scale_factor

        # Calculate new corners of the box using temporary variables
        x1_new_temp = center_x - new_width / 2.0
        y1_new_temp = center_y - new_height / 2.0
        x2_new_temp = center_x + new_width / 2.0
        y2_new_temp = center_y + new_height / 2.0

        # Check if the new coordinates are out of image bounds and discard if so
        if x1_new_temp < 0 or y1_new_temp < 0 or x2_new_temp > image.width or y2_new_temp > image.height:
            #print("New bounding box is out of image bounds")
            return None

        # If within bounds, assign the new coordinates
        x1_new = x1_new_temp
        y1_new = y1_new_temp
        x2_new = x2_new_temp
        y2_new = y2_new_temp

        if (x2_new - x1_new) < 32 or (y2_new - y1_new) < 32:  # Check if the dimensions of the box are too small
            #print("bbox found but is small")
            return None

        # Crop and return the image within the adjusted box
        cropped_image = image.crop((x1_new, y1_new, x2_new, y2_new))
        x_center_norm, y_center_norm, width_norm, height_norm = convert_bbox_coco2yolo(image.width, image.height,(x1,y1,width,height))
        class_name = image_path.split('/')[-2]
        annotation = f"{class_index[class_name]} {x_center_norm} {y_center_norm} {width_norm} {height_norm}\n"

        return (cropped_image, image, annotation)
    
def snake_case(s):
    """Convert a string to snake_case."""
    return '_'.join(s.split()).lower()

def process_folder(input_folder, output_folder, config, retain_uncropped = False):

    models = {
        'FCOS': detection.fcos_resnet50_fpn(weights=detection.FCOS_ResNet50_FPN_Weights.DEFAULT),
        'RetinaNet': detection.retinanet_resnet50_fpn(weights=detection.RetinaNet_ResNet50_FPN_Weights.DEFAULT),
        'SSD': detection.ssd300_vgg16(weights=detection.SSD300_VGG16_Weights.DEFAULT),
        'MaskRCNN': detection.maskrcnn_resnet50_fpn(weights=detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT),
        'FasterRCNN': detection.fasterrcnn_resnet50_fpn(weights=detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT),    
    }
    
    # Load the pre-trained model and set to evaluation mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    # Ensure the output folder exists
    seed_image_folder = os.path.join(os.path.dirname(output_folder),'images')
    os.makedirs(seed_image_folder, exist_ok=True)

    if retain_uncropped:
        real_seeded_image_folder = seed_image_folder.replace('seed_unsplit','real_unsplit')
        real_seeded_label_folder = real_seeded_image_folder.replace('images','labels')
        os.makedirs(real_seeded_image_folder, exist_ok=True)
        os.makedirs(real_seeded_label_folder, exist_ok=True)
    
    # Extract the last folder name from output_folder and convert it to snake_case
    last_folder_name = os.path.basename(os.path.normpath(output_folder))
    last_folder_name_snake_case = snake_case(last_folder_name)  # Assuming snake_case is a defined function

    
    # Process a folder of images, detecting vehicles and saving the images
    for image_file in tqdm(sorted(os.listdir(input_folder))):
        
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, image_file)
            # Use the modified detect_vehicles_and_crop function
            images = detect_vehicles_and_crop(image_path, models, config, device)

            if images is not None:
                cropped_image, image, annotation = images
                # Generate a unique filename with a timestamp
                timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')[:17]
                output_file_name = f"{last_folder_name_snake_case}_{timestamp}.jpg"
                #output_file_name = f"{image_file}"
                output_file_path = os.path.join(seed_image_folder, output_file_name)
                
                # Save the cropped image
                cropped_image.save(output_file_path)
                
                # Save the real images which are used for seed
                if retain_uncropped:
                    image.save(os.path.join(real_seeded_image_folder, output_file_name))
                    annotation_path = os.path.join(real_seeded_label_folder, f"{os.path.splitext(output_file_name)[0]}.txt")
                    with open(annotation_path, 'w') as file:
                        file.write(annotation)
