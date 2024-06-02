import json
import random
from pathlib import Path

import numpy as np
import torch
import transformers
from diffusers import StableDiffusionXLPipeline
from transformers import Owlv2Processor, Owlv2ForObjectDetection

from data_generation import words_bank


def main():
    sdxl_pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/sdxl-turbo", use_safetensors=True
    ).to("cuda")

    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

    bria_pipe = transformers.pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)

    out_dir = Path(f'datasets/generated/union_data')
    out_dir.mkdir(exist_ok=True, parents=True)

    # Generated images that did not meet the criteria
    out_dir_leftovers = Path(f'datasets/generated/union_data_leftovers')
    out_dir_leftovers.mkdir(exist_ok=True, parents=True)

    with open('assets/openimages_classes.txt', 'r') as f:
        objects = f.read().splitlines()

    base_image = None
    saved_base = False
    for _ in range(100000):
        if base_image is not None and not saved_base:
            alternative_path = out_dir_leftovers / f'{object_name.replace(" ", "_")}_{placement.replace(" ", "_")}_{seed}.jpg'
            base_image.save(alternative_path)
        saved_base = False
        try:
            current_objects = []
            object_count = 2
            for _ in range(object_count):
                object_name = random.choice(objects)
                # Remove '/' or any non english characters
                object_name = ''.join(char if char.isalnum() else ' ' for char in object_name)
                # Restrict to two words
                object_name = ' '.join(object_name.split()[:2])
                current_objects.append(object_name)
            object_name = ' and a '.join(current_objects)
            print(object_name)
            placement = random.choice(words_bank.placements) if random.random() < 0.5 else ''

            prompt = f"A photo of {object_name} {placement}"
            seed = random.randint(0, 1000000)

            base_image = sdxl_pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
            object_name = object_name[:50]
            out_path = out_dir / f'{object_name.replace(" ", "_")}_{placement.replace(" ", "_")}_{seed}.jpg'

            # Try to detect the objects in the generated image
            texts = [[f"a {obj}" for obj in current_objects]]
            inputs = processor(text=texts, images=base_image, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)

            # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
            target_sizes = torch.Tensor([base_image.size[::-1]])
            # Convert outputs (bounding boxes and class logits) to Pascal VOC Format (xmin, ymin, xmax, ymax)
            results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes,
                                                              threshold=0.2)

            i = 0  # Retrieve predictions for the first image for the corresponding text queries
            boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

            chosen_boxes = []
            # Take highest scoring box for each label
            for obj_ind in range(len(current_objects)):
                relevant_boxes = boxes[labels == obj_ind]
                relevant_scores = scores[labels == obj_ind]
                if len(relevant_boxes) > 0:
                    max_score_idx = relevant_scores.argmax()
                    max_box = relevant_boxes[max_score_idx]
                    if relevant_scores[max_score_idx] < 0.2:
                        break
                    chosen_boxes.append(max_box)

            # Filter to box with max score
            if len(chosen_boxes) != 2:
                print(f'Skipping, detected {len(chosen_boxes)} objects')
                continue

            # Verify small overlap between two using IoU
            box1, box2 = chosen_boxes
            box1 = box1.int()
            box2 = box2.int()
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - intersection
            iou = float(intersection / union)
            if iou > 0.1:
                print(f'Skipping, iou is {iou}')
                continue

            metadata = {'objects': [], 'placement': placement, 'prompt': prompt}
            masked_objects = []
            for obj_ind, box in enumerate(chosen_boxes):
                box = box.int()
                obj_name = current_objects[obj_ind]
                box_dict = {
                    "x1": int(box[0].item()),
                    "y1": int(box[1].item()),
                    "x2": int(box[2].item()),
                    "y2": int(box[3].item()),
                    "name": obj_name
                }
                object_size = (box_dict['x2'] - box_dict['x1']) * (box_dict['y2'] - box_dict['y1'])
                if object_size < 5000:
                    print(f'Skipping, object size is {object_size}')
                    continue
                metadata['objects'].append(box_dict)
                crop = base_image.crop((box_dict['x1'], box_dict['y1'], box_dict['x2'], box_dict['y2']))
                tmp_path = 'tmp.jpg'
                crop.save(tmp_path)
                with torch.no_grad():
                    masked_object = bria_pipe(tmp_path)
                # Returns RGBA, take mask channel and check how many pixels
                crop_mask = np.array(masked_object)[..., 3]

                # Make sure at least 10000 pixels are non-masked
                # Check if over half of the pixels seen
                total_pixels = crop_mask.size
                seen_pixels = np.sum(crop_mask > 200)
                if seen_pixels / total_pixels < 0.1:
                    print(f'Skipping, not enough pixels seen. only {seen_pixels / total_pixels:.3f} seen')
                    continue
                masked_objects.append(masked_object)
            if len(masked_objects) != len(chosen_boxes):
                continue
            for obj_ind, masked_object in enumerate(masked_objects):
                masked_object.save(str(out_path).replace('.jpg', f'_OBJ_{obj_ind}.png'))
            with open(out_path.with_suffix('.json'), 'w') as f:
                json.dump(metadata, f)

            base_image.save(out_path)
            saved_base = True

        except ValueError as e:
            print(f'Error: {e}')


if __name__ == "__main__":
    main()
