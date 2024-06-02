import json
import random
import traceback
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from data_generation import words_bank

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose(
        [
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


class TexturedDataset(Dataset):
    def __init__(
            self,
            dataset_dir: Path,
            multi_dirs: bool = True,
            textures_dir: Optional[Path] = None,
            randoms_dir: Optional[Path] = None,
            clip_image_size: int = 224,
            image_processor=None,
            tokenizer=None
    ):
        """ Texturing dataset
        Args:
            dataset_dir (Path): Path to the directory containing images.
            multi_dirs (bool): Whether to expect multiple directories in dataset_dir.
            textures_dir (Optional[Path]): Path to the directory containing textures, used for validation
            randoms_dir (Optional[Path]): Path to the directory containing random images, to further diversify the dataset.
            clip_image_size (int): Size of the image to be fed into the model.
            image_processor: ImageProcessor: Image processor to be used.
            tokenizer: Tokenizer: Tokenizer to be used.
        """
        if multi_dirs:
            self.paths = [list(d.glob('*.jpg')) for d in dataset_dir.iterdir() if d.is_dir()]
        else:
            # Read each directory in dataset_dir
            self.paths = [list(dataset_dir.glob('*.jpg'))]
        self.flat_inds = []
        for i, path_list in enumerate(self.paths):
            self.flat_inds.extend([i] * len(path_list))

        print(f'For dataset {dataset_dir}')
        print(f'Loaded {len(self.flat_inds)} images.')
        if textures_dir:
            self.textures = list(textures_dir.glob('*.jpg'))
            print(f'Loaded {len(self.textures)} textures.')
        else:
            self.textures = None

        if randoms_dir:
            self.randoms = list(randoms_dir.glob('*.jpg'))
            print(f'Loaded {len(self.randoms)} randoms.')
        else:
            self.randoms = None
        self.tokenizer = tokenizer
        self.transform1 = _transform(clip_image_size)

        self.clip_image_size = clip_image_size

        self.image_processor = image_processor


    def __len__(self):
        return len(self.flat_inds)

    def crop_patch(self, image, box, mask):
        """ Logic for cropping an exemplar patch from an image. """

        x_min = box[0]
        x_max = box[2]
        y_min = box[1]
        y_max = box[3]
        max_rect_size = min(x_max - x_min, y_max - y_min) // 2
        min_rect_size = min(96, max_rect_size - 1)
        rect_size = random.randint(min_rect_size, max_rect_size)

        base_crop = image.crop(box)
        # Erode mask so we don't choose center from boundary
        mask = np.array(mask)
        mask[mask < 250] = 0
        mask[mask >= 250] = 255

        if np.count_nonzero(mask) == 0:
            x_center = (x_min + x_max) // 2
            y_center = (y_min + y_max) // 2
            out_crop = image.crop((x_center - rect_size // 2, y_center - rect_size // 2, x_center + rect_size // 2,
                                   y_center + rect_size // 2))
        else:
            eroded_mask = cv2.erode(mask, np.ones((min_rect_size + 1, min_rect_size + 1), np.uint8), iterations=1,
                                    borderType=cv2.BORDER_CONSTANT, borderValue=0)
            foreground_y, foreground_x = np.where(eroded_mask == 255)
            foreground_pixels = list(zip(foreground_y, foreground_x))
            if len(foreground_pixels) == 0:
                last_non_empty_mask = mask.copy()
                curr_mask = mask.copy()
                kernel = np.ones((11, 11), np.uint8)
                while np.count_nonzero(curr_mask) > 0:
                    last_non_empty_mask = curr_mask.copy()
                    curr_mask = cv2.erode(curr_mask, kernel, iterations=1, borderType=cv2.BORDER_CONSTANT,
                                          borderValue=0)
                foreground_y, foreground_x = np.where(last_non_empty_mask == 255)
                foreground_pixels = list(zip(foreground_y, foreground_x))

            y_center, x_center = random.choice(foreground_pixels)
            # Check if rect_size is all in mask, if not shorten
            curr_size = rect_size
            # Bound to size of image based on y_center and x_center
            curr_size = min(curr_size, min(x_center, y_center, image.width - x_center, image.height - y_center) * 2)

            while curr_size > min_rect_size - 1:
                if np.all(mask[y_center - curr_size // 2:y_center + curr_size // 2,
                          x_center - curr_size // 2:x_center + curr_size // 2] == 255):
                    break
                curr_size -= 1

            out_crop = base_crop.crop((x_center - curr_size // 2, y_center - curr_size // 2, x_center + curr_size // 2,
                                       y_center + curr_size // 2))

        return out_crop

    def __getitem__(self, i: int):
        is_random = False
        try:
            # If a random directory is given load a random image with some probability
            if self.randoms is not None and random.random() < 0.2:
                random_path = random.choice(self.randoms)
                random_image = Image.open(random_path).convert("RGB")
                source_image = random_image

                target_image = random_image
                # Calculate random crop size of scale 0.1 to 0.5
                rand_scale = random.uniform(0.2, 0.4)
                crop_size = min(target_image.width, target_image.height) * rand_scale
                # Randomly choose crop location around image center, shift by 0.1width max
                crop_center_x = random.randint(int(target_image.width * 0.45), int(target_image.width * 0.55))
                crop_center_y = random.randint(int(target_image.height * 0.45), int(target_image.height * 0.55))
                target_patch = target_image.crop((crop_center_x - crop_size // 2, crop_center_y - crop_size // 2,
                                                  crop_center_x + crop_size // 2, crop_center_y + crop_size // 2))
                is_random = True
            else:
                relevant_paths = self.paths[self.flat_inds[i]]

                target_path = random.choice(relevant_paths)
                if random.random() < 0.2:
                    source_path = target_path
                else:
                    source_path = random.choice(relevant_paths)

                target_image = Image.open(target_path).convert("RGB")
                source_image = Image.open(source_path).convert("RGB")

                # For validation where textures are taken from a different directory
                if self.textures is not None:
                    texture_path = random.choice(self.textures)
                    target_patch = Image.open(texture_path).convert("RGB")
                    target_image = source_image
                else:
                    box_path = target_path.parent / 'box.json'
                    if box_path.exists():
                        with open(box_path, 'r') as f:
                            box = json.load(f)
                            box = (box['x1'], box['y1'], box['x2'], box['y2'])
                    else:
                        box = (0, 0, target_image.width, target_image.height)
                    mask_path = target_path.parent / 'mask.png'
                    if mask_path.exists():
                        mask = Image.open(mask_path).convert("L")
                    else:
                        mask = Image.new('L', (target_image.width, target_image.height), 0)
                    target_patch = self.crop_patch(target_image, box, mask)

            target_image = self.image_processor(target_image)['pixel_values'][0]
            source_image = self.image_processor(source_image)['pixel_values'][0]
            target_patch = self.image_processor(target_patch)['pixel_values'][0]
        except Exception as e:
            print(f'Error processing image: {e}')
            traceback.print_exc()
            empty_image = Image.new('RGB', (self.clip_image_size, self.clip_image_size), (255, 255, 255))
            target_image = self.image_processor(empty_image)['pixel_values'][0]
            source_image = self.image_processor(empty_image)['pixel_values'][0]
            target_patch = self.image_processor(empty_image)['pixel_values'][0]

        out_dict = {}

        text = ""

        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        out_dict["input_ids"] = text_inputs.input_ids[0]
        out_dict["mask"] = text_inputs.attention_mask.bool()[0]
        out_dict["text"] = text

        out_dict["source_a"] = source_image
        out_dict["source_b"] = target_patch
        out_dict["is_random"] = is_random

        return target_image, out_dict


class InstructDataset(Dataset):
    def __init__(
            self,
            dataset_dir: Path,
            clip_image_size: int = 224,
            image_processor=None,
            tokenizer=None,

    ):
        self.paths = list(dataset_dir.glob('*.jpg'))
        self.tokenizer = tokenizer
        self.transform1 = _transform(clip_image_size)

        self.clip_image_size = clip_image_size

        self.image_processor = image_processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i: int):
        path = self.paths[i]
        prompt = ' '.join(path.stem.split('_')[:-1])
        image = Image.open(path).convert("RGB")
        clip_image = self.image_processor(image)['pixel_values'][0]

        out_dict = {}

        if random.random() < 0.5:
            adjective = random.choice(words_bank.adjectives)
            text = adjective
            text_condition = f'A {adjective} {prompt}'
        else:
            art_type = random.choice(words_bank.art_types)
            text = art_type
            text_condition = f'A {art_type} of a {prompt}'

        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        out_dict["input_ids"] = text_inputs.input_ids[0]

        out_dict["mask"] = text_inputs.attention_mask.bool()[0]
        out_dict["text"] = text

        text_condition_inputs = self.tokenizer(
            text_condition,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        out_dict["condition_ids"] = text_condition_inputs.input_ids[0]

        out_dict["text_condition"] = text_condition

        out_dict["source_a"] = clip_image

        return clip_image, out_dict


class PairsDataset(Dataset):
    """ Used for validation, simply creates pairs of images from the same directory."""
    def __init__(
            self,
            dataset_dir: Path,
            clip_image_size: int = 224,
            image_processor=None,
            tokenizer=None,

    ):
        self.paths = list(dataset_dir.glob('*.jpg'))
        self.tokenizer = tokenizer
        self.transform1 = _transform(clip_image_size)

        self.clip_image_size = clip_image_size

        self.image_processor = image_processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i: int):
        path = self.paths[i]
        path_b = random.choice(self.paths)
        image = Image.open(path).convert("RGB")
        image_b = Image.open(path_b).convert("RGB")
        clip_image = self.image_processor(image)['pixel_values'][0]

        clip_image_a = self.image_processor(image)['pixel_values'][0]
        clip_image_b = self.image_processor(image_b)['pixel_values'][0]

        out_dict = {}

        text = ''

        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        out_dict["input_ids"] = text_inputs.input_ids[0]

        out_dict["mask"] = text_inputs.attention_mask.bool()[0]
        out_dict["text"] = text

        out_dict["source_a"] = clip_image_a
        out_dict["source_b"] = clip_image_b

        return clip_image, out_dict


class SceneDataset(Dataset):
    def __init__(
            self,
            dataset_dir: Path,
            background_dir: Path,
            random_dir: Path,
            clip_image_size: int = 224,
            image_processor=None,
            tokenizer=None,

    ):
        self.paths = [p for p in list(dataset_dir.glob('*.jpg')) if not p.name.endswith('_inpainted.jpg')]
        self.backgrounds = list(background_dir.glob('*.jpg'))
        self.randoms = list(random_dir.glob('*.jpg')) if random_dir is not None else None
        self.tokenizer = tokenizer
        self.transform1 = _transform(clip_image_size)

        self.clip_image_size = clip_image_size

        self.image_processor = image_processor

    def __len__(self):
        return len(self.paths)

    def crop_patch(self, image, mask):

        # if mask is empty return zeros
        if np.count_nonzero(mask) == 0:
            return Image.new('RGB', (self.clip_image_size, self.clip_image_size), (255, 255, 255))
        last_non_empty_mask = mask.copy()
        curr_mask = mask.copy()
        kernel = np.ones((11, 11), np.uint8)
        rect_size = 11
        while np.count_nonzero(curr_mask) > 1000:
            rect_size += 11
            last_non_empty_mask = curr_mask.copy()
            curr_mask = cv2.erode(curr_mask, kernel, iterations=1, borderType=cv2.BORDER_CONSTANT,
                                  borderValue=0)
        foreground_y, foreground_x = np.where(last_non_empty_mask == 1)
        foreground_pixels = list(zip(foreground_y, foreground_x))
        y_center, x_center = random.choice(foreground_pixels)
        # Check if rect_size is all in mask, if not shorten
        curr_size = rect_size
        # Bound to size of image based on y_center and x_center
        curr_size = min(curr_size, min(x_center, y_center, image.width - x_center, image.height - y_center) * 2)
        out_crop = image.crop((x_center - curr_size // 2, y_center - curr_size // 2, x_center + curr_size // 2,
                               y_center + curr_size // 2))

        return out_crop

    def __getitem__(self, i: int):
        is_random = False
        try:
            if self.randoms is not None and random.random() < 0.1:
                path = random.choice(self.randoms)
                image = Image.open(path).convert("RGB")
                object_with_background = Image.open(path).convert("RGBA")
                background_patch = Image.open(path).convert("RGBA")
                is_random = True
            else:

                path = self.paths[i]
                image = Image.open(path).convert("RGB")
                if path.with_suffix('.npy').exists():
                    mask = np.load(path.with_suffix('.npy'))  # [:,:,3]
                else:
                    mask = np.load(path.with_name(path.stem + '_inpainted.npy'))
                if random.random() < 0.5:
                    # White background
                    background = Image.new("RGB", image.size, (255, 255, 255))
                else:
                    background_path = random.choice(self.backgrounds)
                    background = Image.open(background_path).convert("RGB")


                background.paste(image, (0, 0), Image.fromarray(mask))
                object_with_background = background

                # Generate a patch from the original background
                # Check if has _inpainted file
                inpainted_path = path.with_name(path.stem + '_inpainted.jpg')
                if inpainted_path.exists():
                    background_patch = Image.open(inpainted_path).convert("RGB")
                else:
                    mask[mask < 10] = 0
                    mask[mask >= 10] = 1
                    mask = 1 - mask
                    background_patch = self.crop_patch(image, mask)
                    # Check size of patch
                    if background_patch.size[0] < 50 or background_patch.size[1] < 50:
                        background_patch = Image.new('RGB', (self.clip_image_size, self.clip_image_size),
                                                     (255, 255, 255))
        except Exception as e:
            print(f'Error processing image: {e}')
            traceback.print_exc()
            empty_image = Image.new('RGB', (self.clip_image_size, self.clip_image_size), (255, 255, 255))
            image = empty_image
            object_with_background = empty_image
            background_patch = empty_image

        clip_image = self.image_processor(image)['pixel_values'][0]
        clip_image_a = self.image_processor(object_with_background.convert('RGB'))['pixel_values'][0]
        try:
            clip_image_b = self.image_processor(background_patch.convert('RGB'))['pixel_values'][0]
        except Exception as e:
            print(f'Error processing image: {e}')
            traceback.print_exc()
            clip_image_b = \
                self.image_processor(Image.new('RGB', (self.clip_image_size, self.clip_image_size), (255, 255, 255)))[
                    'pixel_values'][0]

        out_dict = {}

        text = ''

        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        out_dict["input_ids"] = text_inputs.input_ids[0]

        out_dict["mask"] = text_inputs.attention_mask.bool()[0]
        out_dict["text"] = text

        out_dict["source_a"] = clip_image_a
        out_dict["source_b"] = clip_image_b
        out_dict["is_random"] = is_random

        return clip_image, out_dict


class UnionDataset(Dataset):
    def __init__(
            self,
            dataset_dir: Path,
            random_dir: Path,
            clip_image_size: int = 224,
            image_processor=None,
            tokenizer=None,

    ):
        self.paths = list(dataset_dir.glob('*.jpg'))
        self.randoms = list(random_dir.glob('*.jpg')) if random_dir is not None else None
        self.tokenizer = tokenizer
        self.transform1 = _transform(clip_image_size)

        self.clip_image_size = clip_image_size

        self.image_processor = image_processor

    def __len__(self):
        return len(self.paths)

    def paste_on_background(self, image, background, min_scale=0.4, max_scale=0.8):
        # Calculate aspect ratio and determine resizing based on the smaller dimension of the background
        aspect_ratio = image.width / image.height
        scale = random.uniform(min_scale, max_scale)
        new_width = int(min(background.width, background.height * aspect_ratio) * scale)
        new_height = int(new_width / aspect_ratio)

        # Resize image and calculate position
        image = image.resize((new_width, new_height), resample=Image.LANCZOS)
        pos_x = random.randint(0, background.width - new_width)
        pos_y = random.randint(0, background.height - new_height)

        # Paste the image using its alpha channel as mask if present
        background.paste(image, (pos_x, pos_y), image if 'A' in image.mode else None)
        return background

    def __getitem__(self, i: int):
        is_random = False
        try:
            if self.randoms is not None and random.random() < 0.1:
                path = random.choice(self.randoms)
                image = Image.open(path).convert("RGB")
                object_a = Image.open(path).convert("RGBA")
                object_b = Image.open(path).convert("RGBA")
                is_random = True
            else:
                path = self.paths[i]
                image = Image.open(path).convert("RGB")
                object_a = Image.open(str(path).replace('.jpg', f'_OBJ_0.png')).convert("RGBA")
                object_b = Image.open(str(path).replace('.jpg', f'_OBJ_1.png')).convert("RGBA")

                # Use white background
                background_a = Image.new("RGB", image.size, (255, 255, 255))
                background_b = Image.new("RGB", image.size, (255, 255, 255))

                # Convert to RGB with white background
                self.paste_on_background(object_a, background_a, min_scale=0.8, max_scale=0.95)
                self.paste_on_background(object_b, background_b, min_scale=0.8, max_scale=0.95)
                object_a, object_b = background_a, background_b

                # Random switch between a,b
                if random.random() < 0.5:
                    object_a, object_b = object_b, object_a
        except Exception as e:
            print(f'Error processing image: {e}')
            traceback.print_exc()
            empty_image = Image.new('RGB', (self.clip_image_size, self.clip_image_size), (255, 255, 255))
            image = empty_image
            object_a = empty_image
            object_b = empty_image

        clip_image = self.image_processor(image)['pixel_values'][0]
        clip_image_a = self.image_processor(object_a.convert('RGB'))['pixel_values'][0]
        clip_image_b = self.image_processor(object_b.convert('RGB'))['pixel_values'][0]

        # clip_image = self.image_processor(image)['pixel_values'][0]

        out_dict = {}

        text = ''

        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        out_dict["input_ids"] = text_inputs.input_ids[0]

        out_dict["mask"] = text_inputs.attention_mask.bool()[0]
        out_dict["text"] = text

        out_dict["source_a"] = clip_image_a
        out_dict["source_b"] = clip_image_b
        out_dict["is_random"] = is_random

        return clip_image, out_dict


class ClothesDataset(Dataset):
    def __init__(
            self,
            clip_image_size: int = 224,
            image_processor=None,
            val_mode=False,
            tokenizer=None,

    ):
        from datasets import load_dataset
        self.dataset = load_dataset("mattmdjaga/human_parsing_dataset")['train']
        self.tokenizer = tokenizer
        self.transform1 = _transform(clip_image_size)

        self.relevant_inds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 17]

        self.clip_image_size = clip_image_size

        self.image_processor = image_processor
        self.val_mode = val_mode
        self.val_size = 100

    def __len__(self):
        if self.val_mode:
            return self.val_size
        else:
            return len(self.dataset) - self.val_size

    def crop_by_mask(self, image, mask):
        image = np.array(image)
        # Apply mask on image
        image = image * mask[:, :, None] + (1 - mask[:, :, None]) * 255
        image = image.astype(np.uint8)
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]

        # Crop the image
        cropped_image = Image.fromarray(image[ymin:ymax + 1, xmin:xmax + 1])
        cropped_mask = mask[ymin:ymax + 1, xmin:xmax + 1]
        return cropped_image, cropped_mask

    def paste_on_background(self, image, background, min_scale=0.4, max_scale=0.8):
        # Calculate aspect ratio and determine resizing based on the smaller dimension of the background
        aspect_ratio = image.width / image.height
        scale = random.uniform(min_scale, max_scale)
        new_width = int(min(background.width, background.height * aspect_ratio) * scale)
        new_height = int(new_width / aspect_ratio)

        # Resize image and calculate position
        image = image.resize((new_width, new_height), resample=Image.LANCZOS)
        pos_x = random.randint(0, background.width - new_width)
        pos_y = random.randint(0, background.height - new_height)

        # Paste the image using its alpha channel as mask if present
        background.paste(image, (pos_x, pos_y), image if 'A' in image.mode else None)
        return background

    def __getitem__(self, i: int):
        if self.val_mode:
            i += len(self.dataset) - self.val_size
        image = self.dataset[i]['image']
        mask = self.dataset[i]['mask']

        # Pad image to rect with white background

        crops = []
        for ind in self.relevant_inds:
            curr_mask = np.array(mask) == ind
            object_with_background = Image.new("RGB", (self.clip_image_size, self.clip_image_size), (255, 255, 255))
            if np.any(curr_mask):
                cropped_object, cropped_mask = self.crop_by_mask(image, curr_mask)
                # Make sure crop is above certain size
                if min(cropped_object.size) < 60:  # or max(cropped_object.size) < 80:
                    continue

                if np.count_nonzero(cropped_mask) < 1800:
                    continue
                self.paste_on_background(cropped_object, object_with_background, min_scale=0.8, max_scale=0.95)

            crops.append(self.image_processor(object_with_background)['pixel_values'][0])
        large_axis = max(image.size)
        rect_image = Image.new('RGB', (large_axis, large_axis), (255, 255, 255))
        # Paste in the middle
        rect_image.paste(image, (large_axis // 2 - self.dataset[i]['image'].size[0] // 2,
                                 large_axis // 2 - self.dataset[i]['image'].size[1] // 2))

        target_clip_image = self.image_processor(rect_image)['pixel_values'][0]

        out_dict = {}
        text = ''

        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        out_dict["input_ids"] = text_inputs.input_ids[0]

        out_dict["mask"] = text_inputs.attention_mask.bool()[0]
        out_dict["text"] = text

        out_dict["crops"] = crops

        return target_clip_image, out_dict
