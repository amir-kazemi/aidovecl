import os
import shutil
from src import detect, outpaint, utils
import matplotlib.pyplot as plt

class_index = {
    'passenger car': 0,
    'utility vehicle': 1,
    'bus': 2,
    'van': 3,
    'pickup truck': 4,
    'heavy truck': 5,
}

CONFIG = {
    "base_path": "./figs/fig2",
    "real": "real",
    "seed": "seed",
    "seed_split": "seed_split",
    "outpainted": "outpainted",
}

def setup_directories():
    for key in ["seed", "seed_split", "outpainted"]:
        path = os.path.join(CONFIG["base_path"], key)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

def generate_seed_images():
    real_path = os.path.join(CONFIG["base_path"], CONFIG["real"])
    seed_path = os.path.join(CONFIG["base_path"], CONFIG["seed"])
    for vehicle_type in os.listdir(real_path):
        input_folder = os.path.join(real_path, vehicle_type)
        output_folder = os.path.join(seed_path, vehicle_type)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        print(vehicle_type)
        detect.process_folder(input_folder, output_folder)

def split_dataset():
    seed_path = os.path.join(CONFIG["base_path"], CONFIG["seed"])
    seed_split_path = os.path.join(CONFIG["base_path"], CONFIG["seed_split"])
    splitter = utils.DatasetSplitter(seed_path, seed_split_path, split_ratios={'train': 1, 'val': 0, 'test': 0})
    splitter.execute()

def perform_outpainting():
    seed_split_path = os.path.join(CONFIG["base_path"], CONFIG["seed_split"])
    outpainted_path = os.path.join(CONFIG["base_path"], CONFIG["outpainted"])
    utils.set_seeds()
    outpaint.generate_outpainted(seed_split_path, outpainted_path, max_iter=100)

def plot_outpainted_images():
    plt.rcdefaults()
    image_loader = utils.ImageLoader(os.path.join(CONFIG["base_path"], CONFIG["outpainted"], 'train', 'images'))
    images = image_loader.load_images_from_folder()
    image_loader.plot_images(images, class_index)

def run_fig():
    setup_directories()
    generate_seed_images()
    split_dataset()
    perform_outpainting()
    plot_outpainted_images()
    
if __name__ == "__main__":
    run_fig()