import random
from pathlib import Path

from diffusers import StableDiffusionXLPipeline

from data_generation import words_bank
from dataclasses import dataclass
import pyrallis


@dataclass
class RunConfig:
    # Generation mode, should be either 'objects' or 'scenes'
    type: str = 'objects'
    out_dir: Path = Path('datasets/generated/random_objects')
    n_images: int = 100000


@pyrallis.wrap()
def generate(cfg: RunConfig):
    sdxl_pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/sdxl-turbo", use_safetensors=True
    ).to("cuda")

    if cfg.type == 'objects':
        generate_objects = True
    elif cfg.type == 'scenes':
        generate_objects = False
    else:
        raise ValueError(f"Invalid type {cfg.type}")

    cfg.out_dir.mkdir(exist_ok=True, parents=True)

    if generate_objects:
        with open('assets/openimages_classes.txt', 'r') as f:
            objects = f.read().splitlines()

    for _ in range(cfg.n_images):
        try:
            placement = random.choice(words_bank.placements) if random.random() < 0.5 else ''
            if cfg.type == 'objects':
                object_name = random.choice(objects)
                object_name = ''.join(char if char.isalnum() else ' ' for char in object_name)
                prompt = f"A photo of a {object_name} {placement}"
            else:
                object_name = ''
                prompt = f"A photo of an empty {placement.split(' ')[-1]}"
            seed = random.randint(0, 1000000)

            base_image = sdxl_pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]

            out_path = cfg.out_dir / f'{object_name.replace(" ", "_")}_{placement.replace(" ", "_")}_{seed}.jpg'
            base_image.save(out_path)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    # Use to generate objects or backgrounds
    generate(generate_objects=True)
