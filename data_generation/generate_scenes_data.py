import random
from pathlib import Path

import numpy as np
import torch
import transformers
from PIL import Image, ImageFilter
from diffusers import StableDiffusionXLPipeline, AutoPipelineForInpainting

from data_generation import words_bank


def main():
    sdxl_pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/sdxl-turbo", use_safetensors=True
    ).to("cuda")

    bria_pipe = transformers.pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)
    inpaint_pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                                                             torch_dtype=torch.float16, variant="fp16").to("cuda")

    with open('assets/openimages_classes.txt', 'r') as f:
        objects = f.read().splitlines()

    out_dir = Path(f'datasets/generated/scenes_data/')
    out_dir.mkdir(exist_ok=True, parents=True)

    for _ in range(100000):
        try:
            object_name = random.choice(objects)
            # Remove special characters
            object_name = ''.join(char if char.isalnum() else ' ' for char in object_name)
            # Restrict to two words
            object_name = ' '.join(object_name.split()[:2])
            placement = random.choice(words_bank.placements) if random.random() < 0.5 else ''

            prompt = f"A photo of {object_name} {placement}"
            seed = random.randint(0, 1000000)

            base_image = sdxl_pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]

            tmp_path = 'tmp.jpg'
            base_image.save(tmp_path)
            crop_mask = bria_pipe(tmp_path, return_mask=True)  # Retuns a PIL mask

            # Dilate mask
            crop_mask_for_inpaint = np.array(crop_mask)
            crop_mask_for_inpaint[crop_mask_for_inpaint > 10] = 255
            crop_mask_for_inpaint[crop_mask_for_inpaint <= 10] = 0
            crop_mask_for_inpaint = Image.fromarray(crop_mask_for_inpaint).filter(ImageFilter.MaxFilter(31))
            crop_mask_for_inpaint = crop_mask_for_inpaint.convert("RGB")

            inpainted_image = inpaint_pipe(
                prompt=f'A photo of empty {placement.split(" ")[-1]}',
                image=base_image,
                mask_image=crop_mask_for_inpaint,
                guidance_scale=8.0,
                num_inference_steps=20,
                strength=1.0,
            ).images[0]

            # Restrict object_name to 50 characters
            object_name = object_name[:50]
            out_path = out_dir / f'{object_name.replace(" ", "_")}_{placement.replace(" ", "_")}_{seed}.jpg'
            base_image.save(out_path)

            out_path = out_dir / f'{object_name.replace(" ", "_")}_{placement.replace(" ", "_")}_{seed}_inpainted.jpg'
            inpainted_image.save(out_path)

            np.save(out_path.with_suffix('.npy'), np.array(crop_mask).astype(np.uint8))


        except Exception as e:
            print(f'Error: {e}')


if __name__ == "__main__":
    main()
