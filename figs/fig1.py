import os
import shutil
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from src import detect, outpaint, backdrop, utils

# Configuration and paths
CONFIG = {
    "base_path": "./figs/fig1",
    "real": "real",
    "seed": "seed",
    "seed_split": "seed_split",
    "outpainted": "outpainted",
    "prompt": "A street during winter."
}

def setup_directories():
    paths = ["seed", "seed_split", "outpainted", "canvas", "mask"]
    for path in paths:
        full_path = os.path.join(CONFIG["base_path"], path)
        if os.path.exists(full_path):
            shutil.rmtree(full_path)
        os.makedirs(full_path)

def generate_seed_images():
    real_path = os.path.join(CONFIG["base_path"], CONFIG["real"])
    for vehicle_type in os.listdir(real_path):
        input_folder = os.path.join(real_path, vehicle_type)
        detect.process_folder(input_folder, input_folder.replace('real', 'seed'))

def split_dataset():
    seed_path = os.path.join(CONFIG["base_path"], CONFIG["seed"])
    seed_split_path = os.path.join(CONFIG["base_path"], CONFIG["seed_split"])
    splitter = utils.DatasetSplitter(seed_path, seed_split_path, split_ratios={'train': 1, 'val': 0, 'test': 0})
    splitter.execute()

def perform_outpainting():
    seed_split_path = os.path.join(CONFIG["base_path"], CONFIG["seed_split"])
    outpainted_path = os.path.join(CONFIG["base_path"], CONFIG["outpainted"])
    utils.set_seeds()
    outpaint.generate_outpainted(seed_split_path, outpainted_path, max_iter=50, prompt=CONFIG["prompt"], save_interim=True)

def get_nth_image_file(directory, n=0):
    files = sorted([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if files and n < len(files):
        return Image.open(os.path.join(directory, files[n]))
    return None

def create_visualizations():
    plt.rcParams['font.size'] = 20         # Default font size
    plt.rcParams['axes.titlesize'] = 18    # Axes title font size
    plt.rcParams['axes.labelsize'] = 16    # X and Y axis label font size
    plt.rcParams['xtick.labelsize'] = 14   # X axis tick labels
    plt.rcParams['ytick.labelsize'] = 14   # Y axis tick labels
    plt.rcParams['legend.fontsize'] = 16   # Legend font size

    # Define directories
    dirs = {
        'real': os.path.join(CONFIG['base_path'], 'real', 'passenger_car'),
        'seed': os.path.join(CONFIG['base_path'], 'seed_split', 'train', 'passenger_car'),
        'canvas': os.path.join(CONFIG['base_path'], 'canvas', 'train', 'images'),
        'mask': os.path.join(CONFIG['base_path'], 'mask', 'train', 'images'),
        'outpainted': os.path.join(CONFIG['base_path'], 'outpainted', 'train', 'images')
    }

    # Load images
    images = {key: get_nth_image_file(path, 0) for key, path in dirs.items()}

    # Create figure and subplots
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))
    fig.subplots_adjust(wspace=0.05)  # Decrease the space between subplots

    titles = ['Real Image', 'Seed Image', 'Canvas', 'Mask', 'Outpainted Image']
    for ax, (title, key) in zip(axs, zip(titles, images.keys())):
        if images[key]:
            ax.imshow(images[key])
            ax.set_title(title)
            ax.axis('off')

        if key in ['canvas', 'mask']:
            # Highlight canvas and mask
            rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes, linewidth=3, edgecolor='black', facecolor='none')
            ax.add_patch(rect)

    # Highlighting canvas and mask
    bbox_canvas = axs[2].get_position()
    bbox_mask = axs[3].get_position()
    x0 = min(bbox_canvas.x0, bbox_mask.x0) - 0.03
    y0 = min(bbox_canvas.y0, bbox_mask.y0) - 0.15
    x1 = max(bbox_canvas.x1, bbox_mask.x1) + 0.045
    y1 = max(bbox_canvas.y1, bbox_mask.y1) + 0.15
    rect_big = Rectangle((x0, y0), x1 - x0, y1 - y0, transform=fig.transFigure, linewidth=3, edgecolor='black', facecolor='none', linestyle='--')
    fig.patches.append(rect_big)
    fig.text((x0 + x1) / 2, y0 + 0.02, '$Prompt: $ ' + CONFIG["prompt"], fontname='monospace', ha='center', va='bottom', color='black')

    # Add arrows between images, except between canvas and mask
    for i in range(4):
        if i != 2:  # Skip arrow between canvas and mask
            axs[i].annotate('', xy=(1.05, 0.5), xycoords='axes fraction', xytext=(1.2, 0.5),
                            arrowprops=dict(arrowstyle="<|-", lw=4, color='red', shrinkA=0, shrinkB=0, mutation_scale=30))

    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['base_path'], "pipeline.pdf"), format="pdf", bbox_inches='tight', dpi=300)
    plt.show()

def run_fig():
    setup_directories()
    generate_seed_images()
    split_dataset()
    perform_outpainting()
    create_visualizations()

if __name__ == "__main__":
    run_fig()