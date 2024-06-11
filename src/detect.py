import os
import torch
from torchvision import models, transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_mobilenet_v3_large_320_fpn #, FasterRCNN_ResNet50_FPN_Weights
from PIL import Image
import datetime
import inspect
import numpy as np
from tqdm import tqdm

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

def detect_vehicles_and_crop(image_path, model, device):
    # Detect vehicles in an image using the provided model
    image = Image.open(image_path)
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    
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
    max_area = 0
    largest_box = None
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
        scale_factor = 1.10

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
        return (cropped_image, image)
    
def snake_case(s):
    """Convert a string to snake_case."""
    return '_'.join(s.split()).lower()

def process_folder(input_folder, output_folder, retain_uncropped = False):
    
    # Load the pre-trained model and set to evaluation mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model = fasterrcnn_resnet50_fpn(pretrained=True)
    model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
    model.to(device).eval()

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    if retain_uncropped: os.makedirs(output_folder.replace('seed','real_ok'), exist_ok=True)
    
    # Extract the last folder name from output_folder and convert it to snake_case
    last_folder_name = os.path.basename(os.path.normpath(output_folder))
    last_folder_name_snake_case = snake_case(last_folder_name)  # Assuming snake_case is a defined function

    
    # Process a folder of images, detecting vehicles and saving the images
    for image_file in tqdm(sorted(os.listdir(input_folder))):
        
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, image_file)
            # Use the modified detect_vehicles_and_crop function
            images = detect_vehicles_and_crop(image_path, model, device)

            if images is not None:
                cropped_image, image = images
                # Generate a unique filename with a timestamp
                timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')[:17]
                output_file_name = f"{last_folder_name_snake_case}_{timestamp}.jpg"
                #output_file_name = f"{image_file}"
                output_file_path = os.path.join(output_folder, output_file_name)
                
                # Save the cropped image
                cropped_image.save(output_file_path)
                
                # Save the real images which are used for seed
                if retain_uncropped: image.save(output_file_path.replace('seed','real_ok'))
