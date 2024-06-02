import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pyrallis
import torch
from PIL import Image
from diffusers import PriorTransformer, UNet2DConditionModel, KandinskyV22Pipeline
from tqdm import tqdm
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor, CLIPTokenizer, CLIPTextModelWithProjection

from model import pops_utils
from model.pipeline_pops import pOpsPipeline
from huggingface_hub import hf_hub_download
from utils import vis_utils
from data_generation import words_bank


@dataclass
class RunConfig:
    # Path to the learned prior in local filesystem or huggingface
    prior_path: Path
    # Input directory
    dir_a: Path
    # The repo to download the prior from, if None, assumes prior_path is a local path
    prior_repo: Optional[str] = None
    output_dir_name: Path = Path('inference/results_instruct')
    # Path to the kandinsky repo
    kandinsky_prior_repo: str = 'kandinsky-community/kandinsky-2-2-prior'
    kandinsky_decoder_repo: str = 'kandinsky-community/kandinsky-2-2-decoder'
    prior_guidance_scale: List[float] = field(default_factory=lambda: [1.0])
    prior_seeds: List[int] = field(default_factory=lambda: [18, 42])
    unet_seeds: List[int] = field(default_factory=lambda: [0, 1])
    texts: List[str] = field(default_factory=lambda: words_bank.adjectives)


@pyrallis.wrap()
def main(cfg: RunConfig):
    output_dir = cfg.output_dir_name  # cfg.prior_path.parent / cfg.output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    weight_dtype = torch.float16
    device = 'cuda:0'
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(cfg.kandinsky_prior_repo,
                                                                  subfolder='image_encoder',
                                                                  torch_dtype=weight_dtype).eval()

    image_processor = CLIPImageProcessor.from_pretrained(cfg.kandinsky_prior_repo,
                                                         subfolder='image_processor')

    tokenizer = CLIPTokenizer.from_pretrained(cfg.kandinsky_prior_repo, subfolder='tokenizer')
    text_encoder = CLIPTextModelWithProjection.from_pretrained(cfg.kandinsky_prior_repo,
                                                               subfolder='text_encoder',
                                                               torch_dtype=weight_dtype).eval().to(device)

    prior = PriorTransformer.from_pretrained(
        cfg.kandinsky_prior_repo, subfolder="prior"
    )

    if cfg.prior_repo:
        # Load from huggingface
        prior_path = hf_hub_download(repo_id=cfg.prior_repo, filename=str(cfg.prior_path))
    else:
        prior_path = cfg.prior_path

    prior_state_dict = torch.load(prior_path, map_location=device)
    msg = prior.load_state_dict(prior_state_dict, strict=False)
    print(msg)

    prior.eval()

    # Freeze text_encoder and image_encoder
    image_encoder.requires_grad_(False)

    # Load full model for vis
    unet = UNet2DConditionModel.from_pretrained(cfg.kandinsky_decoder_repo,
                                                subfolder='unet').to(torch.float16).to(device)
    prior_pipeline = pOpsPipeline.from_pretrained(cfg.kandinsky_prior_repo,
                                                  prior=prior,
                                                  image_encoder=image_encoder,
                                                  torch_dtype=torch.float16)
    prior_pipeline = prior_pipeline.to(device)
    prior = prior.to(weight_dtype)
    decoder = KandinskyV22Pipeline.from_pretrained(cfg.kandinsky_decoder_repo, unet=unet,
                                                   torch_dtype=torch.float16)
    decoder = decoder.to(device)

    # glob for both jpgs or pths
    inputs_a = [path for path in cfg.dir_a.glob('*.jpg')] + [path for path in cfg.dir_a.glob('*.pth')]

    paths = [(input_a, text) for input_a in inputs_a for text in cfg.texts]

    # just so we have more variety to look at during the inference
    random.shuffle(paths)

    for input_a_path, text in tqdm(paths):
        def process_image(input_path):
            image_caption_suffix = ''
            if input_path is not None and input_path.suffix == '.pth':
                image = torch.load(input_path).to(device).to(weight_dtype)
                embs_unnormed = (image * prior.clip_std) + prior.clip_mean
                zero_embeds = prior_pipeline.get_zero_embed(embs_unnormed.shape[0], device=embs_unnormed.device)
                direct_from_emb = decoder(image_embeds=embs_unnormed, negative_image_embeds=zero_embeds,
                                          num_inference_steps=50, height=512,
                                          width=512, guidance_scale=4).images
                image_pil = direct_from_emb[0]
                image_caption_suffix = '(embedding)'
            else:
                if input_path is not None:
                    image_pil = Image.open(input_path).convert("RGB").resize((512, 512))
                else:
                    image_pil = Image.new('RGB', (512, 512), (255, 255, 255))

                image = torch.Tensor(image_processor(image_pil)['pixel_values'][0]).to(device).unsqueeze(0).to(
                    weight_dtype)

            return image, image_pil, image_caption_suffix

        # Process both inputs
        image_a, image_pil_a, caption_suffix_a = process_image(input_a_path)

        text_inputs = tokenizer(
            text,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        mask = text_inputs.attention_mask.bool()  # [0]

        text_encoder_output = text_encoder(text_inputs.input_ids.to(device))
        text_encoder_hidden_states = text_encoder_output.last_hidden_state
        text_encoder_concat = text_encoder_hidden_states[:, :mask.sum().item()]
        #

        input_image_embeds, input_hidden_state = pops_utils.preprocess(image_a, None,
                                                           image_encoder,
                                                           prior.clip_mean.detach(), prior.clip_std.detach(),
                                                           concat_hidden_states=text_encoder_concat)

        input_images = [image_pil_a]
        captions = [f'{text}{caption_suffix_a}']

        out_name = f"{input_a_path.stem if input_a_path is not None else ''}_{text}"
        for seed in cfg.prior_seeds:
            negative_input_embeds = torch.zeros_like(input_image_embeds)
            negative_hidden_states = torch.zeros_like(input_hidden_state)
            for scale in cfg.prior_guidance_scale:
                img_emb = prior_pipeline(input_embeds=input_image_embeds, input_hidden_states=input_hidden_state,
                                         negative_input_embeds=negative_input_embeds,
                                         negative_input_hidden_states=negative_hidden_states,
                                         num_inference_steps=25,
                                         num_images_per_prompt=1,
                                         guidance_scale=scale,
                                         generator=torch.Generator(device=device).manual_seed(seed))
                torch.save(img_emb, output_dir / f"{out_name}_s_{seed}_cfg_{scale}_img_emb.pth")
                negative_emb = img_emb.negative_image_embeds
                for seed_2 in cfg.unet_seeds:
                    images = decoder(image_embeds=img_emb.image_embeds, negative_image_embeds=negative_emb,
                                     num_inference_steps=50, height=512,
                                     width=512, guidance_scale=4,
                                     generator=torch.Generator(device=device).manual_seed(seed_2)).images
                    input_images += images
                    captions.append(f"prior_s {seed}, cfg {scale} unet_s {seed_2}")  # , ")
        gen_images = vis_utils.create_table_plot(images=input_images, captions=captions)

        gen_images.save(output_dir / f"{out_name}.jpg")
    print('Done!')


if __name__ == "__main__":
    main()
