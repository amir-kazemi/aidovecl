from diffusers import AutoPipelineForText2Image
import torch
import datetime
import random
import os
from tqdm import tqdm

def generate_background(base_folder, sample_size):


    prompts = [
        "A deserted city street at sunrise",
        "An empty urban freeway during twilight",
        "A vacant city highway at dusk",
        "An abandoned road with streetlights flickering at night",
        "A vacant urban alleyway lit by neon lights",
        "An uninhabited boulevard with tall buildings on either side",
        "A deserted overpass with views of a sleeping city",
        "An empty suburban street lined with trees and street lamps",
        "A vacant downtown avenue during a foggy morning",
        "An empty bridge over a city river at night",
        "A vacant pedestrian path in an urban park at dawn",
        "A deserted railway crossing in an urban setting",
        "An empty roundabout with a modern art sculpture",
        "A vacant parking garage in an urban area during sunset",
        "A deserted bike lane along a cityscape",
        "An empty city square at midnight",
        "A vacant pedestrian bridge in the early morning",
        "An empty plaza with fountains not running",
        "A deserted urban staircase leading to a metro station",
        "An empty courtyard surrounded by high-rise buildings",
        "A vacant market street with closed stalls",
        "A deserted promenade along the waterfront",
        "An empty intersection in a major city at dawn",
        "A vacant scenic lookout over a metropolitan area",
        "An empty footbridge with no pedestrians",
        "A deserted highway ramp during early morning",
        "A vacant shopping district at sunrise",
        "An empty boulevard with decorative lights",
        "A deserted dockside walkway at night",
        "An empty urban trail through a park",
        "A vacant main street in a downtown area",
        "A deserted canal path with no boats",
        "An empty waterfront area with benches",
        "A vacant tourist spot in the off-season",
        "An empty festival street with banners still hanging",
        "A deserted outdoor amphitheater",
        "A vacant cultural district during a cloudy day",
        "An empty rooftop park overlooking the city",
        "A deserted boardwalk early in the morning",
        "An empty opera square at dusk",
        "A vacant street with historic architecture",
        "An empty urban bridge during a foggy night",
        "A deserted garden path in an urban area",
        "An empty cathedral square without visitors",
        "A vacant viewing area of a city skyline",
        "An empty pathway in a botanical garden",
        "A deserted street with murals on buildings",
        "An empty boulevard with autumn leaves",
        "A vacant esplanade during a sunset",
        "An empty urban passage under a bridge",
        "A deserted picnic area in a city park",
        "An empty scenic drive with overlooks",
        "A vacant town square with a clock tower",
        "An empty mews in a historical district",
        "A deserted urban waterfall without tourists",
        "An empty street lined with modern sculptures",
        "A vacant avenue during a misty morning",
        "An empty historic site with cobblestone paths",
        "A deserted look-out point at a national park within a city",
        "An empty urban courtyard with sculptures",
        "A vacant pedestrian zone in a business district",
        "An empty scenic route with panoramic views of the city",
        "A deserted skywalk connecting buildings",
        "An empty street in a newly developed area",
        "A vacant dock with no ships",
        "An empty urban island with recreational areas",
        "A deserted cul-de-sac in a residential area",
        "An empty atrium in a modern building",
        "A vacant promenade next to urban architecture",
        "An empty pathway along a riverbank",
        "A deserted road with a view of skyscrapers",
        "An empty square with a historic monument",
        "A vacant urban gazebo in a park",
        "An empty street during a citywide event",
        "A deserted path in a public garden",
        "An empty road leading to a city landmark",
        "A vacant outdoor gallery in an art district",
        "An empty urban trail with skyline views",
        "A deserted pedestrian mall during the evening",
        "An empty boardwalk with closed shops",
        "A vacant fairground after hours",
        "An empty walkway in a high-tech zone",
        "A deserted street under construction",
        "An empty tunnel road in an urban area",
        "A vacant street during a holiday morning",
        "An empty lane in a financial district",
        "A deserted pathway through a redeveloped area",
        "An empty riverside walk with urban art",
        "A vacant street during early morning fog",
        "An empty stairway in a busy district",
        "A deserted urban viaduct at dawn",
        "An empty lot in a bustling area",
        "A deserted shopping arcade during a quiet morning",
        "An empty harborfront walkway with distant city lights",
        "A vacant historic alley with cobblestone flooring at night",
        "An empty urban park with modern art installations",
        "A deserted scenic corridor between high-rise buildings",
        "An empty concrete plaza surrounded by governmental buildings",
        "A vacant overground walkway connecting urban structures",
        "An empty ceremonial court in front of a public building",
    ]

    
    negative_prompt = "car, vehicle, bus, truck, van, aerial view"

    pipeline = AutoPipelineForText2Image.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16
    ).to("cuda")
    pipeline.set_progress_bar_config(disable=True)
    
    for _ in tqdm(range(sample_size)):
        
        prompt = random.choice(prompts)
        
        background_image = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
        ).images[0]


        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')[:17]
        background_file_name = f"background_{timestamp}.jpg"
        os.makedirs(base_folder, exist_ok=True)
        background_file_path = os.path.join(base_folder, background_file_name)
        background_image.save(background_file_path)