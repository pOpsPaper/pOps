# pOps: Photo-Inspired Diffusion Operators

 
> Elad Richardson, Yuval Alaluf, Ali Mahdavi-Amiri, Daniel Cohen-Or
> 
> Text-guided image generation enables the creation of visual content from textual descriptions. 
However, certain visual concepts cannot be effectively conveyed through language alone.  This has sparked a renewed interest in utilizing the CLIP image embedding space for more visually-oriented tasks through methods such as IP-Adapter. Interestingly, the CLIP image embedding space has been shown to be semantically meaningful, where linear operations within this space yield semantically meaningful results. Yet, the specific meaning of these operations can vary unpredictably across different images.
To harness this potential, we introduce pOps, a framework that trains specific semantic operators directly on CLIP image embeddings. 
Each pOps operator is built upon a pretrained Diffusion Prior model. 
While the Diffusion Prior model was originally trained to map between text embeddings and image embeddings, we demonstrate that it can be tuned to accommodate new input conditions, resulting in a diffusion operator.
Working directly over image embeddings not only improves our ability to learn semantic operations but also allows us to directly use a textual CLIP loss as an additional supervision when needed.
We show that pOps can be used to learn a variety of photo-inspired operators with distinct semantic meanings, highlighting the semantic diversity and potential of our proposed approach.


<a href="https://arxiv.org/abs/2406.01300"><img src="https://img.shields.io/badge/arXiv-2406.01300-b31b1b.svg" height=20.5></a>
<a href="https://popspaper.github.io/pOps/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=red" height=20.5></a> 
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/pOpsPaper/pOps-space)



<p align="center">
<img src="https://popspaper.github.io/pOps/static/figures/teaser_pops.jpg" width="800px"/>  
<br>
Different operators trained using pOps. Our method learns operators that are applied directly in the image embedding space, resulting in a variety of semantic operations that can then be realized as images using an image diffusion model.
</p>


## Description :scroll:
Official implementation of the paper "pOps: Photo-Inspired Diffusion Operators"

## Getting started with pOps :rocket:
To set up the environment with all necessary dependencies, please run:
```
pip install -r requirements.txt
```


## Inference 🧑‍🎨



We provide pretrained models for our different operators under an huggingface model card.

### Binary Operators

To run a binary operator, simply use the `scripts.infer_binary` script with the corresponding config file.

```bash
python -m scripts.infer_binary --config_path=configs/infer/texturing.yaml
# or
python -m scripts.infer_binary --config_path=configs/infer/union.yaml
# or
python -m scripts.infer_binary --config_path=configs/infer/scene.yaml
```

This will automatically download the pretrained model and run the inference on the default input images.

Configuration is managed by pyrallis, some useful flags to use with the `scripts.infer_binary` script are:
- `--output_dir_name`: The name of the output directory where the results will be saved.
- `--dir_a`: The path to the directory containing the input images for the first input.
- `--dir_b`: The path to the directory containing the input images for the second input.
- `--vis_mean`: Show results of the mean of the two inputs.

For compositions of multiple operators note that the inference script outputs both the resulting images and the corresponding clip embeddings.
Thus, you can simply feed a directory of embeddings to either `dir_a` or `dir_b`. Useful filtering flags are:
- `--file_exts_a` (/b): Filter to only `.jpg` images or `.pth` embeddings.
- `--name_filter_a` (/b): Filter to only images with specific names.

To sample results with missing input conditions, use the `--drop_condition_a` or `--drop_condition_b` flags.

Finally, to use the IP-Adapter with the inference script, use the `--use_ipadapter` flag and to use additional depth conditioning, use the `--use_depth` flag.

### Instruct Operator

To run the instruct operator, use the `scripts.infer_instruct` script with the corresponding config file.

```bash
python -m scripts.infer_instruct --config_path=configs/infer/instruct.yaml
```

## Training 📉

### Data Generation
We provide several scripts for data generation under the `data_generation` directory.
- `generate_textures.py`: Generates textures data.
- `generate_scenes.py`: Generates scenes data.
- `generate_unions.py`: Generates unions data.

The scene operator also requires random backgrounds which can be generated using the `generate_random_images.py` script.
```bash
python -m data_generation.generate_random_images --output_dir=datasets/random_backgrounds --type=scenes
```

The `generate_random_images.py` script can also be used to generate random images for the other operators
```bash
python -m data_generation.generate_random_images --output_dir=datasets/random_images --type=objects
```

These images can be used for the unconditional steps in training, as will be described in the training section.

### Training Script
Training itself is managed by the `scripts.train` script. See the `configs/training` directory for the different training configurations.

```bash
python -m scripts.train --config_path=configs/training/texturing.yaml
# or 
python -m scripts.train --config_path=configs/training/scene.yaml
# or
python -m scripts.train --config_path=configs/training/union.yaml
# or
python -m scripts.train --config_path=configs/training/instruct.yaml
# or
python -m scripts.train --config_path=configs/training/clothes.yaml
```

The operator itself is defined via the `--mode` flag, which can be set to the specific operator.

Relevant data paths and validation paths can be set in the configuration file.

Use the optional `randoms_dir` flag to specify the directory of random images for the unconditional steps.

## Acknowledgements
Our codebase heavily relies on the [Kandinsky model](https://github.com/ai-forever/Kandinsky-2)

## Citation
If you use this code for your research, please cite the following paper:
```
Coming soon
```
