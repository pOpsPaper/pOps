import json
import random
from pathlib import Path

import torch
import transformers
from diffusers import StableDiffusionDepth2ImgPipeline, StableDiffusionXLPipeline
from transformers import Owlv2Processor, Owlv2ForObjectDetection

from data_generation import words_bank


def main():
    sdxl_pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/sdxl-turbo", use_safetensors=True
    ).to("cuda")

    pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-depth", torch_dtype=torch.float16, ).to("cuda")

    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

    bria_pipe = transformers.pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)

    out_dir = Path(f'datasets/generated/texture_data')
    out_dir.mkdir(exist_ok=True, parents=True)

    with open('assets/openimages_classes.txt', 'r') as f:
       objects = f.read().splitlines()

    for _ in range(100000):
        object_name = random.choice(objects)
        # Remove special characters
        object_name = ''.join(char if char.isalnum() else ' ' for char in object_name)
        # Restrict to two words
        object_name = ' '.join(object_name.split()[:2])
        placement = random.choice(words_bank.placements) if random.random() < 0.5 else ''
        prompt = f"A {object_name} {placement}"

        seed = random.randint(0, 1000000)
        object_out_dir = out_dir / f'{object_name.replace(" ", "_")}_{placement.replace(" ", "_")}_{seed}'
        if object_out_dir.exists():
            continue
        object_out_dir.mkdir(exist_ok=True, parents=True)

        base_image = sdxl_pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
        out_path = object_out_dir / f'base.jpg'
        base_image.save(out_path)

        # Find box
        texts = [[f"a {object_name}"]]
        inputs = processor(text=texts, images=base_image, return_tensors="pt")
        outputs = model(**inputs)

        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor([base_image.size[::-1]])
        # Convert outputs (bounding boxes and class logits) to Pascal VOC Format (xmin, ymin, xmax, ymax)
        results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)

        i = 0  # Retrieve predictions for the first image for the corresponding text queries
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
        # Filter to box with max score
        if len(boxes) == 0:
            scores = torch.tensor([1.0])
            boxes = torch.tensor([[0, 0, base_image.size[0], base_image.size[1]]])

        max_score_idx = scores.argmax()
        box = boxes[max_score_idx]

        # Save box to json file
        box_dict = {
            "x1": int(box[0].item()),
            "y1": int(box[1].item()),
            "x2": int(box[2].item()),
            "y2": int(box[3].item()),
        }
        crop = base_image.crop((box_dict['x1'], box_dict['y1'], box_dict['x2'], box_dict['y2']))
        tmp_path = 'tmp.jpg'
        crop.save(tmp_path)
        crop_mask = bria_pipe(tmp_path, return_mask=True)
        crop_mask.save(object_out_dir / 'mask.png')
        with open(object_out_dir / 'box.json', 'w') as f:
            json.dump(box_dict, f)

        for _ in range(5):
            num_samples = random.randint(1, 5)
            sample_attributes = random.sample(words_bank.texture_attributes, num_samples)
            prompt = f"A {object_name} made from {' '.join(sample_attributes)} {placement}"
            n_propmt = "bad, deformed, ugly, bad anotomy"
            seed = random.randint(0, 1000000)
            image = pipe(prompt=prompt, image=base_image, negative_prompt=n_propmt, strength=1.0,
                         generator=torch.Generator().manual_seed(seed)).images[0]
            attrs_str = '_'.join(sample_attributes)[:100]
            out_path = object_out_dir / f'{attrs_str}_{seed}.jpg'
            image.save(out_path)


if __name__ == "__main__":
    main()
