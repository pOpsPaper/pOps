import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pyrallis
import torch
from PIL import Image
from diffusers import PriorTransformer, UNet2DConditionModel, KandinskyV22Pipeline, AutoPipelineForText2Image
from tqdm import tqdm
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

from model.pipeline_pops import pOpsPipeline
from model import pops_utils
from utils import vis_utils
from huggingface_hub import hf_hub_download
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
import numpy as np

@dataclass
class RunConfig:
    # Path to the learned prior in local filesystem or huggingface
    prior_path: Path
    # Input A directory
    dir_a: Path
    # Input B directory
    dir_b: Path
    # The repo to download the prior from, if None, assumes prior_path is a local path
    prior_repo: Optional[str] = None
    # Path to the kandinsky repo
    kandinsky_prior_repo: str = 'kandinsky-community/kandinsky-2-2-prior'
    kandinsky_decoder_repo: str = 'kandinsky-community/kandinsky-2-2-decoder'
    # For naming the output images
    input_a_name: str = 'objects'
    input_b_name: str = 'textures'
    # Output directory
    output_dir_name: Path = Path('inference/binary_results')
    prior_guidance_scale: List[float] = field(default_factory=lambda: [1.0])
    prior_seeds: List[int] = field(default_factory=lambda: [18, 42])
    unet_seeds: List[int] = field(default_factory=lambda: [0, 1])
    # Look for pths to load embeddings
    file_exts_a: List[str] = field(default_factory=lambda: ['.jpg', '.pth'])
    file_exts_b: List[str] = field(default_factory=lambda: ['.jpg', '.pth'])
    # Look for only specific names in that directory, useful for creating compositions with specific images
    name_filter_a: Optional[List[str]] = None
    name_filter_b: Optional[List[str]] = None
    # Whether to drop the condition for a or b
    drop_condition_a: bool = False
    drop_condition_b: bool = False
    vis_mean: bool = False
    # Whether to also vis with ip adapter
    use_ipadapter: bool = False
    # Whether to use depth conditioning
    use_depth: bool = False


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

    # Optionally load ip adapter
    if cfg.use_ipadapter:
        ip_pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                                torch_dtype=torch.float16).to("cuda")
        ip_pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
        ip_pipeline.set_ip_adapter_scale(1.0)

    if cfg.use_depth:
        depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
        feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0",
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
        ).to("cuda")
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")
        depth_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            vae=vae,
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
        ).to("cuda")
        depth_pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
        depth_pipe.set_ip_adapter_scale(1.0)
        depth_pipe.enable_model_cpu_offload()

        def get_depth_map(image):
            image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
            with torch.no_grad(), torch.autocast("cuda"):
                depth_map = depth_estimator(image).predicted_depth

            depth_map = torch.nn.functional.interpolate(
                depth_map.unsqueeze(1),
                size=(1024, 1024),
                mode="bicubic",
                align_corners=False,
            )
            depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
            depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
            image = torch.cat([depth_map] * 3, dim=1)

            image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
            image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
            return image

    name_is_valid = lambda name, filters: filters is None or any([filter in name for filter in filters])

    # glob for both jpgs or pths
    if cfg.dir_a.is_dir():
        inputs_a = []
        for ext in cfg.file_exts_a:
            inputs_a += [path for path in cfg.dir_a.glob(f'*{ext}') if name_is_valid(path.stem, cfg.name_filter_a)]
    else:
        inputs_a = [cfg.dir_a]

    if cfg.dir_b.is_dir():
        inputs_b = []
        for ext in cfg.file_exts_b:
            inputs_b += [path for path in cfg.dir_b.glob(f'*{ext}') if name_is_valid(path.stem, cfg.name_filter_b)]
    else:
        inputs_b = [cfg.dir_b]

    paths = [(input_a, input_b) for input_a in inputs_a for input_b in inputs_b]

    # just so we have more variety to look at during the inference
    random.seed(42)
    random.shuffle(paths)

    for input_a_path, input_b_path in tqdm(paths):
        def process_image(input_path):
            image_caption_suffix = ''
            if input_path is not None and input_path.suffix == '.pth':
                image = torch.load(input_path).image_embeds.to(device).to(weight_dtype)
                embs_unnormed = image
                zero_embeds = prior_pipeline.get_zero_embed(embs_unnormed.shape[0], device=embs_unnormed.device)
                direct_from_emb = decoder(image_embeds=embs_unnormed, negative_image_embeds=zero_embeds,
                                          num_inference_steps=50, height=512,
                                          width=512, guidance_scale=4,
                                          generator=torch.Generator(device=device).manual_seed(0)).images
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
        image_b, image_pil_b, caption_suffix_b = process_image(input_b_path)
        should_drop_cond = [(cfg.drop_condition_a, cfg.drop_condition_b)]
        input_image_embeds, input_hidden_state = pops_utils.preprocess(image_a, image_b,
                                                                       image_encoder,
                                                                       prior.clip_mean.detach(),
                                                                       prior.clip_std.detach(),
                                                                       should_drop_cond=should_drop_cond)

        input_images = [image_pil_a, image_pil_b]

        if cfg.use_depth:
            depth_image = get_depth_map(image_pil_a)

        captions = [f"{cfg.input_a_name}{caption_suffix_a}", f"{cfg.input_b_name}{caption_suffix_b}"]

        out_name = f"{input_a_path.stem if input_a_path is not None else ''}_{input_b_path.stem if input_b_path is not None else ''}"
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
                    captions.append(f"prior_s {seed}, cfg {scale} unet_s {seed_2}")

                    if cfg.use_ipadapter:
                        images = ip_pipeline(
                            prompt="",
                            ip_adapter_image_embeds=[
                                torch.stack([torch.zeros_like(img_emb.image_embeds), img_emb.image_embeds])],
                            negative_prompt="deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality",
                            num_inference_steps=50,
                            generator=torch.Generator(device="cuda").manual_seed(seed_2),
                        ).images

                        input_images += [images[0].resize((512, 512))]
                        captions.append(f"prior_s {seed}, cfg {scale}, unet_s {seed_2} IP-Adapter")
                    if cfg.use_depth:
                        images = depth_pipe(
                            prompt="",
                            image=depth_image,
                            ip_adapter_image_embeds=[
                                torch.stack([torch.zeros_like(img_emb.image_embeds), img_emb.image_embeds])],
                            negative_prompt="",
                            num_inference_steps=50,
                            controlnet_conditioning_scale=0.5,
                            generator=torch.Generator(device="cuda").manual_seed(seed_2),
                        ).images

                        input_images += [images[0].resize((512, 512))]
                        captions.append(
                            f"prior_s {seed}, cfg {scale}, unet_s {seed_2} IP-Adapter Depth")  #, unet_s {seed_2}")

        if cfg.vis_mean:
            mean_emb = 0.5 * input_hidden_state[:, 0] + 0.5 * input_hidden_state[:, 1]
            mean_emb = (mean_emb * prior.clip_std) + prior.clip_mean
            zero_embeds = prior_pipeline.get_zero_embed(mean_emb.shape[0], device=mean_emb.device)
            direct_from_emb = decoder(image_embeds=mean_emb, negative_image_embeds=zero_embeds,
                                      num_inference_steps=50, height=512,
                                      width=512, guidance_scale=4,
                                      generator=torch.Generator(device=device).manual_seed(seed_2)).images
            input_images += direct_from_emb
            captions.append("mean")
        gen_images = vis_utils.create_table_plot(images=input_images, captions=captions)

        gen_images.save(output_dir / f"{out_name}.jpg")
    print('Done!')


if __name__ == "__main__":
    main()
