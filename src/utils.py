import os
import shutil
import random
from PIL import Image

import random
import numpy as np
import torch

import re
import matplotlib.pyplot as plt

import piq
from skimage.metrics import structural_similarity as ssim

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
        random.shuffle(files)
        n = len(files)

        indices = [int(self.split_ratios['val'] * n), int((self.split_ratios['val'] + self.split_ratios['test']) * n)]
        val_files, test_files, train_files = files[:indices[0]], files[indices[0]:indices[1]], files[indices[1]:]

        for file_list, split in zip([train_files, val_files, test_files], ['train', 'val', 'test']):
            self.copy_files(file_list, split, folder_path)

    def copy_files(self, files, split, folder_path):
        for file in files:
            src = os.path.join(folder_path, file)
            if self.is_image_file(src):
                dst_folder = os.path.join(self.destination_root, split, os.path.basename(folder_path))
                os.makedirs(dst_folder, exist_ok=True)
                shutil.copy(src, dst_folder)

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

    def calculate_ssim(self, image1, image2, mask):
        # Convert PIL images to numpy arrays directly without converting to grayscale
        array1 = np.array(image1)
        array2 = np.array(image2)
        mask_array = np.array(mask)

        # Ensure mask is binary (0s and 1s), and prepare it for RGB by repeating it across three channels
        mask_binary = mask_array[:, :, 0] // 255  # Take one channel (assuming RGB), and convert to binary
        mask_3channel = np.stack([mask_binary] * 3, axis=-1)  # Repeat mask across RGB channels

        # Invert the mask: areas to match are black (0s), so we invert to make them white (1s)
        inverted_mask = 1 - mask_3channel

        # Apply the inverted mask
        masked1 = array1 * inverted_mask
        masked2 = array2 * inverted_mask

        # Calculate SSIM over the masked region. SSIM function can handle multichannel data
        score = ssim(masked1, masked2, multichannel=True)
        return score

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
    
# plotting samples of classified images
class ImageLoader:
    def __init__(self, folder_path):
        self.folder_path = folder_path

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

    def plot_images(self, images, class_index):
        num_classes = len(class_index)
        max_images = max(len(img_list) for img_list in images.values())
        fig, axs = plt.subplots(max_images, num_classes, figsize=(num_classes * 2, max_images * 2))
        if num_classes == 1:
            axs = [axs]
        axs = [axs[i] if num_classes > 1 else [axs[i]] for i in range(max_images)]
        for class_name, col_index in class_index.items():
            if class_name in images:
                for row_index, img in enumerate(images[class_name]):
                    axs[row_index][col_index].imshow(img)
                    axs[row_index][col_index].axis('off')
                    if row_index == 0:
                        axs[row_index][col_index].set_title(class_name)
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.savefig("./figs/fig2/outpainted.pdf", format="pdf", bbox_inches='tight', dpi=300)

