from pathlib import Path

import transformers
from PIL import Image


def main():
    # base_image = Image.open("assets/template_images/mug.png")
    bria_pipe = transformers.pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)

    source_dir = Path('assets/objects')
    out_dir = Path(f'assets/objects_no_back')
    out_dir.mkdir(exist_ok=True, parents=True)

    for path in source_dir.glob('*.jpg'):
        try:
            base_image = Image.open(path).resize((512, 512))
            tmp_path = 'tmp.jpg'
            base_image.save(tmp_path)
            crop_mask = bria_pipe(tmp_path, return_mask=True) # Retuns a PIL mask
            # Apply mask on base_image
            base_image = Image.composite(base_image, Image.new('RGB', base_image.size, (255, 255, 255)), crop_mask)
            base_image.save(out_dir / path.name)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    main()
