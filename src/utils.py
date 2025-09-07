import os
import shutil
from PIL import Image

import random
import numpy as np
import torch
import torchvision.transforms as transforms
import re
import matplotlib.pyplot as plt

import piq

from math import pi

import yaml

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt

# Ultralytics augment imports
from ultralytics.data.augment import Mosaic, MixUp, Compose, LetterBox, Format
from ultralytics.utils.instance import Instances

# Load existing configuration
with open('config/aidovecl-config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Access the class_index dictionary
class_index = config['class_index']


    
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
    def __init__(self, source_root, destination_root, split_ratios={'train': 0.8, 'val': 0.1, 'test': 0.1}, seed=None):
        self.source_root = source_root
        self.destination_root = destination_root
        self.split_ratios = split_ratios
        self.validate_ratios()
        self.prepare_directories()
        self.seed = seed

    def validate_ratios(self):
        total = sum(self.split_ratios.values())
        if total != 1.0:
            raise ValueError("The sum of split ratios must be 1.")

    def prepare_directories(self):
        for split in self.split_ratios.keys():
            os.makedirs(os.path.join(self.destination_root, split), exist_ok=True)

    def split_and_copy_files(self, folder_path, files):
        if self.seed is not None:
            random.seed(self.seed)
            
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
        
        # Initialize models and move them to the specified device
        self.clip_iqa = piq.CLIPIQA().to(self.device)
        self.qualiclip = torch.hub.load(repo_or_dir="miccunifi/QualiCLIP", source="github", model="QualiCLIP").eval().to(device)

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

    def calculate_qualiclip(self, image):
        # Define the preprocessing pipeline
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])
        # Preprocess the image
        image = preprocess(image).unsqueeze(0).to(self.device)
        # Compute the quality score
        with torch.no_grad(), torch.cuda.amp.autocast():
            qualiclip_score = self.qualiclip(image).item()
        return qualiclip_score


    
# plotting samples of classified images
class ImageLoader:
    def __init__(self, folder_path,config):
        self.folder_path = folder_path
        self.class_index = config['class_index']

    def load_images_from_folder(self):
        images = {}
        for filename in sorted(os.listdir(self.folder_path)):
            if filename.endswith((".png", ".jpg", ".jpeg")):
                class_name = re.split(r'\d', filename)[0].rstrip('_')
                class_name = class_name.replace('_', ' ')
                img_path = os.path.join(self.folder_path, filename)
                img = Image.open(img_path)
                if class_name in images:
                    images[class_name].append(img)
                else:
                    images[class_name] = [img]
        return images

    def plot_images(self, images):
        num_classes = len(self.class_index)
        max_images = max(len(img_list) for img_list in images.values())
        fig, axs = plt.subplots(num_classes, max_images, figsize=(max_images * 1.5, num_classes * 1.5))
        if num_classes == 1:
            axs = [axs]
        axs = [axs[j] if max_images > 1 else [axs[j]] for j in range(num_classes)]
        for class_name, row_index in self.class_index.items():
            if class_name in images:
                for col_index, img in enumerate(images[class_name]):
                    axs[row_index][col_index].imshow(img)
                    axs[row_index][col_index].axis('off')
                    if col_index == 0:
                        axs[row_index][col_index].set_title(class_name, loc='left', x=-0.15, y =.3, rotation=90)
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.savefig("./figs/fig3/outpainted.pdf", format="pdf", bbox_inches='tight', dpi=300)
        

def plot_radar_chart(data):
    """
    Plots a radar chart with different line styles for each model and a custom category order.

    Parameters:
    - data (dict): Dictionary with categories as keys and dictionaries of model votes as values.

    Example input:
    {
        'category1': {'model1': value, 'model2': value, ...},
        'category2': {'model1': value, 'model2': value, ...},
        ...
    }
    """
    # Normalize category names for lookup and ordering
    category_order = sorted(data.keys(), key=lambda x: class_index[x.lower()])

    # Extract categories and models
    categories = category_order
    models = ['FCOS', 'RetinaNet', 'SSD', 'MaskRCNN', 'FasterRCNN']  # Ensure legend order

    # Reformat data for radar chart
    votes = []
    for model in models:
        votes.append([data[category][model] for category in categories])

    # Convert to numpy array for easier handling
    votes = np.array(votes)

    # Number of categories (subfolders)
    num_categories = len(categories)
    angles = [n / float(num_categories) * 2 * pi for n in range(num_categories)]
    angles += angles[:1]  # Complete the circle

    # Line styles
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]

    # Plot radar chart
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    # Plot each model
    for i, model_votes in enumerate(votes):
        values = model_votes.tolist()
        values += values[:1]  # Repeat the first value to close the circle
        ax.plot(angles, values, label=models[i], linestyle=line_styles[i])
        ax.fill(angles, values, alpha=0.1)  # Optional: Fill area under the lines

    # Add labels and legend
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([category.capitalize() for category in categories], fontsize=10)  # Capitalized labels with specific font size
    ax.set_rlabel_position(0)
    max_value = max(max(votes.flatten()), 1)  # Dynamically set max value

    plt.title("Consensus Level", fontsize=17, position=(0.5, 1.1), ha='center')  # Title with specific font size

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), title="Models", fontsize=14)  # Legend with specific font size

    ax.set_yticks([0.1, 0.2, 0.3, 0.4])
    ax.tick_params(axis='both', which='major', labelsize=14)  # Set font size for all tick labels

    #plt.tight_layout()
    os.makedirs("./figs/fig2", exist_ok=True)
    plt.savefig("./figs/fig2/consensus_level.pdf", format="pdf", bbox_inches='tight', dpi=300)
    





