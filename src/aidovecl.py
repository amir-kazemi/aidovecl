import os
import shutil
from src import detect, outpaint, utils, backdrop
import matplotlib.pyplot as plt
from huggingface_hub import login
# Use your Hugging Face API token
#login(token="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")

import yaml

# Load existing configuration
with open('config/aidovecl-config.yaml', 'r') as file:
    config = yaml.safe_load(file)

base_path = "./datasets"

def setup_directories():
    for folder_name in ["seed_unsplit", "seed", "real_unsplit", "real"]:#, "outpainted"]:
        path = os.path.join(base_path, folder_name)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

def generate_seed_images(real_raw_folder_name):
    real_path = os.path.join(base_path, real_raw_folder_name)
    seed_path = os.path.join(base_path, "seed_unsplit")
    for vehicle_type in os.listdir(real_path):
        input_folder = os.path.join(real_path, vehicle_type)
        output_folder = os.path.join(seed_path, vehicle_type)
        print(vehicle_type)
        detect.process_folder(input_folder, output_folder, config, retain_uncropped = True)

def split_dataset(folder_name, seed):
    folder_path = os.path.join(base_path, folder_name)
    folder_split_path = os.path.join(base_path, '_'.join(folder_name.split('_')[:-1]))
    splitter = utils.DatasetSplitter(folder_path, folder_split_path, split_ratios={'train': 0.4, 'val': 0.1, 'test': 0.50}, seed=seed)
    splitter.execute()
    shutil.rmtree(folder_path)

def perform_outpainting():
    seed_split_path = os.path.join(base_path, "seed")
    outpainted_path = os.path.join(base_path, "outpainted")
    outpaint.generate_outpainted(seed_split_path, outpainted_path, config, random_seed = 0)

def add_background(base_folder):
    for subfolder in os.listdir(base_folder):
        subfolder_path = os.path.join(base_folder, subfolder, 'images')
        if os.path.isdir(subfolder_path):
            file_count = len([f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f))])
            file_count = int(file_count*0.1)
            print(f"{subfolder}: {file_count} files")
            backdrop.generate_background(subfolder_path, file_count)


def perform_augmentation():
    """
    Merges the contents of multiple source folders into a single destination folder.
    
    Args:
    source_folders (list of str): List of paths to the source folders.
    destination_folder (str): Path to the destination folder where contents are merged.
    """
    real_seed_split_path = os.path.join(base_path, "real")
    source_folders = [real_seed_split_path]
    source_folders.append(os.path.join(base_path, "outpainted"))
    '''
    for i in range(3):
        outpainted_path = os.path.join(base_path, f"outpainted{i}")
        source_folders.append(outpainted_path)
    '''
    destination_folder = os.path.join(base_path, "augmented")

    def copy_tree(src, dst):
        """
        Recursively copies contents of the source directory to the destination directory.
        
        Args:
        src (str): Path to the source directory.
        dst (str): Path to the destination directory.
        """
        for item in os.listdir(src):
            src_item = os.path.join(src, item)
            dst_item = os.path.join(dst, item)

            if os.path.isdir(src_item):
                if not os.path.exists(dst_item):
                    os.makedirs(dst_item)
                copy_tree(src_item, dst_item)
            else:
                shutil.copy2(src_item, dst_item)

    # Ensure the destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for folder in source_folders:
        if os.path.exists(folder) and os.path.isdir(folder):
            copy_tree(folder, destination_folder)
        else:
            print(f"Warning: {folder} is not a valid directory or does not exist.")

def run():
    
    '''
    setup_directories()

    generate_seed_images("real_raw")

    
    split_dataset("real_unsplit", 0)

    split_dataset("seed_unsplit", 0)

    
    perform_outpainting()

    utils.set_seeds(0)
    add_background(os.path.join(base_path, 'real'))

    utils.set_seeds(1)
    add_background(os.path.join(base_path, 'outpainted'))
    '''
    
    perform_augmentation()
    
    
if __name__ == "__main__":
    run()