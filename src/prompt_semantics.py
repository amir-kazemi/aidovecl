import os
import pandas as pd
import torch
from PIL import Image
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
)
from sentence_transformers import SentenceTransformer, util
from tqdm.auto import tqdm

class ImageCaption:
    """
    A unified class for generating image captions using BLIP and ViT-GPT2, and computing caption similarity.
    """

    def __init__(self):
        # Initialize BLIP model and processor
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

        # Initialize ViT-GPT2 model, feature extractor, and tokenizer
        self.vitgpt2_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.vit_feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.vit_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

        # Initialize Sentence Transformer model for computing similarity
        self.sent_model = SentenceTransformer('all-MiniLM-L6-v2')

    def generate_blip_caption(self, image: Image.Image) -> str:
        """
        Generate a caption for the given image using BLIP.
        """
        inputs = self.blip_processor(image, return_tensors="pt")
        output_ids = self.blip_model.generate(**inputs)
        caption = self.blip_processor.decode(output_ids[0], skip_special_tokens=True)
        return caption

    def generate_vit_caption(self, image: Image.Image, max_length: int = 16, num_beams: int = 4) -> str:
        """
        Generate a caption for the given image using ViT-GPT2.
        """
        inputs = self.vit_feature_extractor(images=image, return_tensors="pt")
        output_ids = self.vitgpt2_model.generate(inputs.pixel_values, max_length=max_length, num_beams=num_beams)
        caption = self.vit_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return caption

    def compute_caption_similarity(self, caption1: str, caption2: str) -> torch.Tensor:
        """
        Compute cosine similarity between two captions using sentence embeddings.
        Returns a tensor containing the similarity score.
        """
        embeddings = self.sent_model.encode([caption1, caption2], convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1])
        return similarity_score

    
class OutpaintedImageLoader:
    """
    Iterate over outpainted image filenames from a CSV located in base_folder,
    locate each file under base_folder/<split_set>/images/, open it with PIL,
    convert to RGB, and return along with prompt.
    """

    def __init__(self, base_folder: str, image_caption_instance):
        """
        Args:
            base_folder (str): Root directory that:
                1. Contains 'outpainted_prompts.csv'
                2. Has subfolders for each split set (e.g., train/, val/, test/),
                   each containing an 'images/' folder
        Raises:
            FileNotFoundError: If the CSV file is not found in base_folder.
        """
        self.base_folder = base_folder
        self.csv_path = os.path.join(self.base_folder, "outpainted_prompts.csv")
        if not os.path.isfile(self.csv_path):
            raise FileNotFoundError(f"CSV file not found at: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        self.ic = image_caption_instance
        
    def insert_vehicle(self, prompt, vehicle):
        words = prompt.split()
        try:
            location = words[4]
        except IndexError:
            # If format unexpected, return prompt unchanged
            return prompt

        # Lowercase vehicle name
        vehicle = vehicle.lower()
        if vehicle == 'pickup':
            vehicle = 'a car (pickup truck)'
        elif vehicle not in ['truck', 'bus', 'minibus']:
            vehicle = f'a car ({vehicle})'
            

        # Lists of 'on' and 'in' locations
        on_locations = [
            "street", "road", "highway", "bridge", "track", "path", "boulevard",
            "avenue", "lane", "trail", "rail", "route", "expressway", "motorway"
        ]

        # Determine preposition
        if location in on_locations:
            preposition = "on"
        else:
            preposition = "in"

        return prompt.replace("of a", f"of {vehicle} {preposition} a", 1)

    def load_all_outpainted(self):
        """
        Iterate through each row of the CSV, build the path to the outpainted image,
        open it with PIL, convert to RGB, compute caption similarities, and save results.

        Returns:
            List of tuples: [(filename, prompt, PIL.Image.Image), ...] for all successfully loaded images.
        """
        loaded_images = []
        records = []  # Collect rows to write to CSV

        for idx, row in tqdm(self.df.iterrows()):
            split = row['split_set']
            filename = row['outpainted_file_name']
            prompt = row['prompt']
            
            vehicle = filename.split('_')[0]
            prompt = self.insert_vehicle(prompt, vehicle)

            # Construct full path: base_folder/split_set/images/filename
            full_path = os.path.join(self.base_folder, split, 'images', filename)

            if not os.path.isfile(full_path):
                print(f"Warning: File not found: {full_path}")
                continue

            # Open with PIL and convert to RGB
            with Image.open(full_path) as img:
                img_rgb = img.convert('RGB')

            # Generate captions and compute similarities
            blip_caption = self.ic.generate_blip_caption(img_rgb)
            vit_caption = self.ic.generate_vit_caption(img_rgb)
            
            blip_prompt_sim = self.ic.compute_caption_similarity(blip_caption, prompt).item()
            vit_prompt_sim = self.ic.compute_caption_similarity(vit_caption, prompt).item()
            blip_vit_sim = self.ic.compute_caption_similarity(blip_caption, vit_caption).item()
            
            # Print or log if desired
            '''
            print(filename, blip_prompt_sim, vit_prompt_sim, blip_vit_sim)
            print(prompt)
            print(blip_caption)
            print(vit_caption)
            '''

            # Append to records
            records.append({
                'split_set': split,
                'outpainted_file_name': filename,
                'prompt': prompt,
                'blip_caption': blip_caption,
                'vit_caption': vit_caption,
                'blip_prompt_sim': blip_prompt_sim,
                'vit_prompt_sim': vit_prompt_sim,
                'blip_vit_sim': blip_vit_sim
            })

        # Create a DataFrame and save to a new CSV in the base_folder
        result_df = pd.DataFrame(records)
        output_csv = os.path.join(self.base_folder, "semantic_similarities.csv")
        result_df.to_csv(output_csv, index=False)

        return loaded_images
    
def save_results():
    ic_instance = ImageCaption()
    oil_instance = OutpaintedImageLoader('datasets/outpainted/', ic_instance)
    oil_instance.load_all_outpainted()