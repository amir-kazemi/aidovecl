import os
import shutil
import random
from PIL import Image
import random
import numpy as np
import torch
import piq

    
def set_seeds(seed_value=0):
    """
    Set seeds for reproducibility across Python's random module, NumPy, and PyTorch.
    
    Args:
    seed_value (int): The seed value to use for all random number generators.
    """
    # Seed Python's random module
    random.seed(seed_value)

    # Seed NumPy's random module
    np.random.seed(seed_value)

    # Seed PyTorch to ensure reproducibility
    torch.manual_seed(seed_value)

    # If using CUDA (GPU), seed all GPUs as well
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        
    # Enforce deterministic behavior in PyTorch when using CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    
    
class DatasetSplitter:
    def __init__(self, source_root, destination_root, split_ratios={'train': 0.8, 'val': 0.1, 'test': 0.1}):
        self.source_root = source_root
        self.destination_root = destination_root
        self.split_ratios = split_ratios
        self.validate_ratios()
        self.prepare_directories()

    def validate_ratios(self):
        total = sum(self.split_ratios.values())
        if total != 1.0:
            raise ValueError("The sum of split ratios must be 1.")

    def prepare_directories(self):
        for split in self.split_ratios.keys():
            os.makedirs(os.path.join(self.destination_root, split), exist_ok=True)

    def split_and_copy_files(self, folder_path, files):
        # Group files by class_name
        classified_files = {}
        for file in files:
            class_name = file.split('_')[0]  # Extract class_name from file name
            if class_name not in classified_files:
                classified_files[class_name] = []
            classified_files[class_name].append(file)
        
        # Initialize lists for train, validation, and test files
        train_files, val_files, test_files = [], [], []
        
        # Perform splitting for each class_name
        for class_name, file_list in classified_files.items():
            random.shuffle(file_list)
            n = len(file_list)

            indices = [int(self.split_ratios['val'] * n), int((self.split_ratios['val'] + self.split_ratios['test']) * n)]
            class_val_files, class_test_files, class_train_files = file_list[:indices[0]], file_list[indices[0]:indices[1]], file_list[indices[1]:]

            val_files.extend(class_val_files)
            test_files.extend(class_test_files)
            train_files.extend(class_train_files)
        
        # Copy files to respective splits
        for file_list, split in zip([train_files, val_files, test_files], ['train', 'val', 'test']):
            self.copy_files(file_list, split, folder_path)

    def copy_files(self, files, split, folder_path):
        dst_image_folder = os.path.join(self.destination_root, split, os.path.basename(folder_path))
        os.makedirs(dst_image_folder, exist_ok=True)
        dst_label_folder = os.path.join(os.path.dirname(dst_image_folder),'labels')
        os.makedirs(dst_label_folder, exist_ok=True)
        for file in files:
            image_src = os.path.join(folder_path, file)
            label_src = os.path.join(os.path.dirname(os.path.dirname(image_src)), 'labels', f"{os.path.splitext(file)[0]}.txt")
            if self.is_image_file(image_src):
                shutil.copy(image_src, dst_image_folder)
                try: shutil.copy(label_src, dst_label_folder)
                except: pass

    def is_image_file(self, filepath):
        try:
            img = Image.open(filepath)
            img.verify()  # Check if it's a valid image
            return True
        except (IOError, SyntaxError):
            return False

    def execute(self):
        for folder in os.listdir(self.source_root):
            folder_path = os.path.join(self.source_root, folder)
            if os.path.isdir(folder_path):
                files = os.listdir(folder_path)
                self.split_and_copy_files(folder_path, files)
                
class ImageAssessment:
    def __init__(self, device=None):
        if device is None or not torch.cuda.is_available():
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize CLIP-IQA model and move it to the specified device
        self.clip_iqa = piq.CLIPIQA().to(self.device)

    def calculate_brisque(self, image):
        # Convert image to grayscale and normalize
        img_gray = np.array(image.convert('L')).astype('float32') / 255.0
        img_tensor = torch.tensor(img_gray).unsqueeze(0).unsqueeze(0)

        # Calculate BRISQUE score and normalize
        brisque_score = piq.brisque(img_tensor, data_range=1.0).item()
        return brisque_score

    def calculate_clip_iqa(self, image):
        # Convert image to RGB and normalize
        img_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_tensor = img_tensor.to(self.device)

        # Calculate CLIP-IQA score using the initialized model
        clip_iqa_score = self.clip_iqa(img_tensor).item()
        return clip_iqa_score

    def calculate_tv_loss(self, image):
        # Convert image to RGB and normalize
        img_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_tensor = img_tensor.to(self.device)

        # Calculate TV loss using PIQ
        tv_loss = piq.TVLoss(reduction='mean')(img_tensor).item()
        return tv_loss

    

        





