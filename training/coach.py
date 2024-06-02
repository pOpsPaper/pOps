import os
import random
import sys
from pathlib import Path

import accelerate
import diffusers
import numpy as np
import pyrallis
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
# matplotlib.use('Agg')  # Set the backend to non-interactive (Agg)
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import PriorTransformer, DDPMScheduler
from diffusers import UNet2DConditionModel, KandinskyV22Pipeline
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.training_utils import EMAModel
from packaging import version
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection
from transformers.utils import ContextManagers

from model.pipeline_pops import pOpsPipeline
from model import pops_utils
from training.dataset import TexturedDataset, InstructDataset, PairsDataset, \
    SceneDataset, UnionDataset, ClothesDataset
from training.train_config import TrainConfig
from utils import vis_utils

logger = get_logger(__name__, log_level="INFO")


class Coach:
    def __init__(self, config: TrainConfig):
        self.cfg = config
        self.cfg.output_dir.mkdir(exist_ok=True, parents=True)
        (self.cfg.output_dir / 'cfg.yaml').write_text(pyrallis.dump(self.cfg))
        (self.cfg.output_dir / 'run.sh').write_text(f'python {Path(__file__).name} {" ".join(sys.argv)}')

        self.logging_dir = self.cfg.output_dir / 'logs'
        accelerator_project_config = ProjectConfiguration(
            total_limit=2, project_dir=self.cfg.output_dir, logging_dir=self.logging_dir
        )
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            mixed_precision=self.cfg.mixed_precision,
            log_with=self.cfg.report_to,
            project_config=accelerator_project_config,
        )
        logger.info(self.accelerator.state, main_process_only=False)
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        if self.cfg.seed is not None:
            set_seed(self.cfg.seed)

        if self.accelerator.is_main_process:
            self.logging_dir.mkdir(exist_ok=True, parents=True)

        self.noise_scheduler = DDPMScheduler(beta_schedule='squaredcos_cap_v2', prediction_type='sample')
        self.image_processor = CLIPImageProcessor.from_pretrained(self.cfg.image_processor_path,
                                                                  subfolder='image_processor')
        self.tokenizer = CLIPTokenizer.from_pretrained(self.cfg.tokenizer_path, subfolder='tokenizer')

        def deepspeed_zero_init_disabled_context_manager():
            """
            returns either a context list that includes one that will disable zero.Init or an empty context list
            """
            deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
            if deepspeed_plugin is None:
                return []

            return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16
        with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.cfg.pretrained_image_encoder,
                                                                               subfolder='image_encoder',
                                                                               torch_dtype=self.weight_dtype).eval()
            self.text_encoder = CLIPTextModelWithProjection.from_pretrained(self.cfg.text_encoder_path,
                                                                            subfolder='text_encoder',
                                                                            torch_dtype=self.weight_dtype).eval()
        print('args.pretrained_prior_path =', self.cfg.pretrained_prior_path)
        self.prior = PriorTransformer.from_pretrained(
            self.cfg.pretrained_prior_path, subfolder="prior"
        )

        self.clip_mean = self.prior.clip_mean.clone()
        self.clip_std = self.prior.clip_std.clone()

        # Freeze text_encoder and image_encoder
        self.text_encoder.requires_grad_(False)
        self.image_encoder.requires_grad_(False)

        # Load full model for vis
        self.unet = UNet2DConditionModel.from_pretrained('kandinsky-community/kandinsky-2-2-decoder',
                                                         subfolder='unet').to(torch.float16).to(self.accelerator.device)
        self.prior_pipeline = pOpsPipeline.from_pretrained(
            'kandinsky-community/kandinsky-2-2-prior',
            prior=self.prior,
            image_encoder=self.image_encoder,
            torch_dtype=torch.float16)
        self.prior_pipeline = self.prior_pipeline.to(self.accelerator.device)
        self.decoder = KandinskyV22Pipeline.from_pretrained('kandinsky-community/kandinsky-2-2-decoder', unet=self.unet,
                                                            torch_dtype=torch.float16)
        self.decoder = self.decoder.to(self.accelerator.device)

        if self.cfg.lora_rank is not None:
            self.prior.requires_grad_(False)
            self.prior.to(self.accelerator.device, dtype=self.weight_dtype)
            lora_attn_procs = {}
            for name in self.prior.attn_processors.keys():
                lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=2048, rank=self.cfg.lora_rank)

            self.prior.set_attn_processor(lora_attn_procs)

            self.lora_layers = AttnProcsLayers(self.prior.attn_processors)

        # Create EMA for the prior.
        if self.cfg.use_ema:
            ema_prior = PriorTransformer.from_pretrained(
                self.cfg.pretrained_prior_path, subfolder="prior"
            )
            ema_prior = EMAModel(ema_prior.parameters(), model_cls=PriorTransformer, model_config=ema_prior.config)
            ema_prior.to(self.accelerator.device)

        # `accelerate` 0.16.0 will have better support for customized saving
        if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
            # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
            def save_model_hook(models, weights, output_dir):
                if self.cfg.use_ema:
                    ema_prior.save_pretrained(os.path.join(output_dir, "prior_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "prior"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

            def load_model_hook(models, input_dir):
                if self.cfg.use_ema:
                    load_model = EMAModel.from_pretrained(os.path.join(input_dir, "prior_ema"), PriorTransformer)
                    ema_prior.load_state_dict(load_model.state_dict())
                    ema_prior.to(self.accelerator.device)
                    del load_model

                for i in range(len(models)):
                    # pop models so that they are not loaded again
                    model = models.pop()

                    # load diffusers style into model
                    load_model = PriorTransformer.from_pretrained(input_dir, subfolder="prior")
                    model.register_to_config(**load_model.config)

                    model.load_state_dict(load_model.state_dict())
                    del load_model

            self.accelerator.register_save_state_pre_hook(save_model_hook)
            self.accelerator.register_load_state_pre_hook(load_model_hook)

        if self.cfg.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        if self.cfg.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
                )

            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        if self.cfg.lora_rank is not None:
            params_to_optimize = list(self.lora_layers.parameters())
        else:
            if self.cfg.training_mode == 'full':
                params_to_optimize = list(self.prior.parameters())
            elif '_layers' in self.cfg.training_mode:
                layers_to_train = int(self.cfg.training_mode.split('_')[0])
                params_to_optimize = []
                self.prior.norm_out.requires_grad_(False)
                self.prior.proj_to_clip_embeddings.requires_grad_(False)
                self.prior.time_proj.requires_grad_(False)
                self.prior.time_embedding.requires_grad_(False)
                self.prior.transformer_blocks[layers_to_train:].requires_grad_(False)
                self.prior.clip_mean.requires_grad_(False)
                self.prior.clip_std.requires_grad_(False)
                # Populate params_to_optimize with all params that require grad and print its name
                for name, param in self.prior.named_parameters():
                    if param.requires_grad:
                        print(name)
                        params_to_optimize.append(param)
            else:
                raise ValueError(f"Invalid training mode: {self.cfg.training_mode}")
                # Count the number of parameters
        num_params = sum(p.numel() for p in params_to_optimize)
        # Pretty print the number of parameters
        print(f"Number of parameters: {num_params}")

        self.optimizer = optimizer_cls(
            params_to_optimize,
            lr=self.cfg.lr,
            betas=(self.cfg.adam_beta1, self.cfg.adam_beta2),
            weight_decay=self.cfg.adam_weight_decay,
            eps=self.cfg.adam_epsilon,
        )

        self.train_dataloader, self.validation_dataloader = self.get_dataloaders()

        if self.cfg.lora_rank is not None:
            self.lora_layers, self.optimizer, self.train_dataloader = self.accelerator.prepare(
                self.lora_layers, self.optimizer, self.train_dataloader
            )
        else:
            self.prior, self.optimizer, self.train_dataloader = self.accelerator.prepare(
                self.prior, self.optimizer, self.train_dataloader
            )

        self.image_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.clip_mean = self.clip_mean.to(self.weight_dtype).to(self.accelerator.device)
        self.clip_std = self.clip_std.to(self.weight_dtype).to(self.accelerator.device)

        self.train_step = 0 if self.cfg.resume_from_step is None else self.cfg.resume_from_step
        print(self.train_step)
        self.device = "cuda"

        if self.cfg.resume_from_path is not None:
            prior_state_dict = torch.load(self.cfg.resume_from_path, map_location=self.device)
            msg = self.prior.load_state_dict(prior_state_dict, strict=False)
            print(msg)

    def save_model(self, save_path):
        save_path.parent.mkdir(exist_ok=True, parents=True)
        prior_state_dict = self.prior.state_dict()
        del prior_state_dict["clip_mean"]
        del prior_state_dict["clip_std"]
        torch.save(prior_state_dict, save_path)

        if self.cfg.lora_rank is not None:
            lora_state_dict = self.lora_layers.state_dict()
            torch.save(lora_state_dict, save_path.parent / "lora.ckpt")


    def unnormalize_and_pil(self, tensor):
        unnormed = tensor * torch.tensor(self.image_processor.image_std).view(3, 1, 1).to(
            tensor.device) + torch.tensor(self.image_processor.image_mean).view(3, 1, 1).to(tensor.device)
        return transforms.ToPILImage()(unnormed)

    def save_images(self, image, conds, input_embeds, hidden_states, target_embeds, label='', save_path=''):
        self.prior.eval()
        input_images = []
        captions = []
        for i in range(len(conds)):
            pil_image = self.unnormalize_and_pil(conds[i]).resize((self.cfg.img_size, self.cfg.img_size))
            # Check if all zeros
            if self.cfg.mode == 'clothes' and np.all(np.array(pil_image) > 250):
                continue
            input_images.append(pil_image)
            captions.append("Condition")
        if image is not None:
            input_images.append(self.unnormalize_and_pil(image).resize((self.cfg.img_size, self.cfg.img_size)))
            captions.append(f"Target {label}")

        seeds = range(2)
        output_images = []
        embebds_to_vis = []
        embeds_captions = []
        if self.cfg.mode not in ['hybrid']:
            embebds_to_vis += [target_embeds]
            embeds_captions += ["Target Reconstruct" if image is not None else "Source Reconstruct"]
        if self.cfg.mode not in ['instruct', 'colorization', 'clothes']:  # ,'histogram']:
            for alpha in [0.5]:
                embebds_to_vis.append(alpha * hidden_states[:, 0] + (1 - alpha) * hidden_states[:, 1])
                embeds_captions.append(f"{alpha:.2f}A + {1 - alpha:.2f}B")
        for embs in embebds_to_vis:
            embs_unnormed = (embs * self.clip_std) + self.clip_mean
            zero_embeds = self.prior_pipeline.get_zero_embed(embs.shape[0], device=embs.device)
            direct_from_emb = self.decoder(image_embeds=embs_unnormed, negative_image_embeds=zero_embeds,
                                           num_inference_steps=50, height=512,
                                           width=512, guidance_scale=4).images
            output_images = output_images + direct_from_emb
        captions += embeds_captions

        for seed in seeds:
            negative_input_embeds = torch.zeros_like(input_embeds)
            negative_hidden_states = torch.zeros_like(hidden_states)
            for scale in [1, 4]:
                img_emb = self.prior_pipeline(input_embeds=input_embeds, input_hidden_states=hidden_states,
                                              negative_input_embeds=negative_input_embeds,
                                              negative_input_hidden_states=negative_hidden_states,
                                              num_inference_steps=25,
                                              num_images_per_prompt=1,
                                              guidance_scale=scale,
                                              generator=torch.Generator(device="cuda").manual_seed(seed))

                negative_emb = img_emb.negative_image_embeds
                for seed_2 in range(1):
                    images = self.decoder(image_embeds=img_emb.image_embeds, negative_image_embeds=negative_emb,
                                          num_inference_steps=50, height=512,
                                          width=512, guidance_scale=4,
                                          generator=torch.Generator(device="cuda").manual_seed(seed_2)).images
                    # torch.Generator(device="cuda").manual_seed(s) for s in range(4)
                    output_images += images
                    captions.append(f"prior_s {seed}, cfg {scale}, unet_s {seed_2}")

        all_images = input_images + output_images
        gen_images = vis_utils.create_table_plot(images=all_images, captions=captions)
        gen_images.save(save_path)
        self.prior.train()

    def get_dataloaders(self) -> torch.utils.data.DataLoader:
        sampler_train = None
        if self.cfg.mode == 'texture':
            dataset_path = self.cfg.dataset_path
            if not isinstance(self.cfg.dataset_path, list):
                dataset_path = [self.cfg.dataset_path]
            datasets = []
            for path in dataset_path:
                datasets.append(TexturedDataset(
                    dataset_dir=path,
                    randoms_dir=self.cfg.randoms_dir,
                    image_processor=self.image_processor,
                    tokenizer=self.tokenizer
                ))
            dataset = torch.utils.data.ConcatDataset(datasets)
            dataset_weights = []
            for single_dataset in datasets:
                dataset_weights.extend([len(dataset) / len(single_dataset)] * len(single_dataset))
            sampler_train = torch.utils.data.WeightedRandomSampler(weights=dataset_weights,
                                                                   num_samples=len(dataset_weights))

            validation_dataset = TexturedDataset(
                dataset_dir=self.cfg.val_dataset_path,
                multi_dirs=False,
                textures_dir=self.cfg.textures_dir,
                image_processor=self.image_processor,
                tokenizer=self.tokenizer
            )
        elif self.cfg.mode == 'instruct':
            dataset = InstructDataset(
                dataset_dir=self.cfg.dataset_path,
                image_processor=self.image_processor,
                tokenizer=self.tokenizer,
            )
            validation_dataset = InstructDataset(dataset_dir=self.cfg.val_dataset_path,
                                                 image_processor=self.image_processor, tokenizer=self.tokenizer)
        elif self.cfg.mode == 'clothes':
            dataset = ClothesDataset(
                image_processor=self.image_processor,
                tokenizer=self.tokenizer,
            )
            validation_dataset = ClothesDataset(
                image_processor=self.image_processor,
                tokenizer=self.tokenizer,
                val_mode=True
            )
        elif self.cfg.mode == 'union':
            dataset_path = self.cfg.dataset_path
            if not isinstance(self.cfg.dataset_path, list):
                dataset_path = [self.cfg.dataset_path]
            datasets = []
            for path in dataset_path:
                datasets.append(UnionDataset(dataset_dir=path,
                                             random_dir=self.cfg.randoms_dir,
                                             image_processor=self.image_processor,
                                             tokenizer=self.tokenizer))
            dataset = torch.utils.data.ConcatDataset(datasets)
            dataset_weights = []
            for single_dataset in datasets:
                dataset_weights.extend([len(dataset) / len(single_dataset)] * len(single_dataset))
            sampler_train = torch.utils.data.WeightedRandomSampler(weights=dataset_weights,
                                                                   num_samples=len(dataset_weights))

            validation_dataset = PairsDataset(dataset_dir=self.cfg.val_dataset_path,
                                              image_processor=self.image_processor, tokenizer=self.tokenizer)
        elif self.cfg.mode == 'scene':
            dataset_path = self.cfg.dataset_path
            if not isinstance(self.cfg.dataset_path, list):
                dataset_path = [self.cfg.dataset_path]
            datasets = []
            for path in dataset_path:
                datasets.append(SceneDataset(
                    dataset_dir=path,
                    background_dir=self.cfg.backgrounds_dir,
                    random_dir=self.cfg.randoms_dir,
                    image_processor=self.image_processor,
                    tokenizer=self.tokenizer
                ))
            dataset = torch.utils.data.ConcatDataset(datasets)
            dataset_weights = []
            for single_dataset in datasets:
                dataset_weights.extend([len(dataset) / len(single_dataset)] * len(single_dataset))
            sampler_train = torch.utils.data.WeightedRandomSampler(weights=dataset_weights,
                                                                   num_samples=len(dataset_weights))
            validation_dataset = TexturedDataset(
                dataset_dir=self.cfg.val_dataset_path,
                multi_dirs=False,
                textures_dir=self.cfg.textures_dir,
                image_processor=self.image_processor,
                tokenizer=self.tokenizer
            )
        else:
            raise ValueError(f"Invalid mode: {self.cfg.mode}")
        train_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.cfg.train_batch_size, shuffle=sampler_train is None,
            num_workers=self.cfg.num_workers,
            sampler=sampler_train
        )

        validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset, batch_size=1, shuffle=True, num_workers=self.cfg.num_workers
        )
        return train_dataloader, validation_dataloader

    def train(self):
        pbar = tqdm(range(self.train_step, self.cfg.max_train_steps + 1))
        while self.train_step < self.cfg.max_train_steps:
            train_loss = 0.0
            self.prior.train()

            for sample_idx, batch in enumerate(self.train_dataloader):
                # self.optimizer.zero_grad()
                with self.accelerator.accumulate(self.prior):
                    image, cond = batch

                    image = image.to(self.weight_dtype).to(self.accelerator.device)
                    if 'source_a' in cond:
                        cond['source_a'] = cond['source_a'].to(self.weight_dtype).to(self.accelerator.device)
                    if 'source_b' in cond:
                        cond['source_b'] = cond['source_b'].to(self.weight_dtype).to(self.accelerator.device)
                    if 'crops' in cond:
                        for crop_ind in range(len(cond['crops'])):
                            cond['crops'][crop_ind] = cond['crops'][crop_ind].to(self.weight_dtype).to(
                                self.accelerator.device)
                    for key in cond.keys():
                        if isinstance(cond[key], torch.Tensor):
                            cond[key] = cond[key].to(self.accelerator.device)

                    with torch.no_grad():
                        text_mask = cond['mask']
                        text_encoder_output = self.text_encoder(cond['input_ids'])
                        text_encoder_hidden_states = text_encoder_output.last_hidden_state
                        # Take only the unmasked tokens
                        text_encoder_concat = None

                        if self.cfg.mode == 'instruct':
                            text_target = self.text_encoder(cond['condition_ids']).text_embeds
                            text_encoder_concat = text_encoder_hidden_states[:, :text_mask.sum().item()]

                        image_embeds = self.image_encoder(image).image_embeds

                        # Sample noise that we'll add to the image_embeds
                        noise = torch.randn_like(image_embeds)
                        bsz = image_embeds.shape[0]
                        # Sample a random timestep for each image
                        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,),
                                                  device=image_embeds.device)
                        timesteps = timesteps.long()
                        image_embeds = (image_embeds - self.clip_mean) / self.clip_std
                        noisy_latents = self.noise_scheduler.add_noise(image_embeds, noise, timesteps)

                        target = image_embeds

                        input_image_embeds, input_hidden_state = self.preprocess_inputs(cond=cond,
                                                                                        text_encoder_concat=text_encoder_concat)

                # Create the input sequence for the prior
                loss = 0
                image_feat_seq = torch.zeros_like(text_encoder_hidden_states)
                image_feat_seq[:, :input_hidden_state.shape[1]] = input_hidden_state
                image_txt_mask = torch.zeros_like(text_mask)
                image_txt_mask[:, :input_hidden_state.shape[1]] = 1

                # Predict the noise residual and compute loss
                model_pred = self.prior(
                    noisy_latents,
                    timestep=timesteps,
                    proj_embedding=input_image_embeds,
                    encoder_hidden_states=image_feat_seq,
                    attention_mask=image_txt_mask,
                ).predicted_image_embedding

                # Calculate the loss
                if self.cfg.mode == 'instruct':
                    prior_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    normed_pred = model_pred.float() * self.clip_std.detach() + self.clip_mean.detach()
                    normed_pred = F.normalize(normed_pred, p=2, dim=-1)
                    normed_text_target = F.normalize(text_target.float(), p=2, dim=-1)
                    clip_loss = 1 - (normed_pred * normed_text_target).sum(dim=-1)
                    prior_loss = prior_loss + self.cfg.clip_strength * clip_loss
                else:
                    prior_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                loss += prior_loss
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = self.accelerator.gather(loss.repeat(self.cfg.train_batch_size)).mean()
                train_loss += avg_loss.item() / self.cfg.gradient_accumulation_steps

                # Backprop
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.prior.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    if self.cfg.use_ema:
                        self.ema_prior.step(self.prior.parameters())
                    pbar.update(1)
                    self.train_step += 1
                    train_loss = 0.0

                    if self.train_step % self.cfg.checkpointing_steps == 1:
                        if self.accelerator.is_main_process:
                            save_path = self.cfg.output_dir / f"learned_prior.pth"
                            self.save_model(save_path)
                            logger.info(f"Saved state to {save_path}")
                    pbar.set_postfix(**{"loss": loss.cpu().detach().item()})

                    if self.cfg.log_image_frequency > 0 and (
                            self.train_step % self.cfg.log_image_frequency == 1) or self.train_step < 10:
                        image_save_path = self.cfg.output_dir / 'images' / f"{self.train_step}_step_images.jpg"
                        image_save_path.parent.mkdir(exist_ok=True, parents=True)
                        # Apply the full diffusion process
                        conds_list = []
                        if 'source_a' in cond:
                            conds_list.append(cond['source_a'][0])
                        if 'source_b' in cond:
                            conds_list.append(cond['source_b'][0])
                        if 'crops' in cond:
                            for crop_ind in range(len(cond['crops'])):
                                conds_list.append(cond['crops'][crop_ind][0])

                        label = ''
                        if 'text_condition' in cond:
                            label += cond['text_condition'][0]
                        self.save_images(image=image[0], conds=conds_list,
                                         label=label,
                                         input_embeds=input_image_embeds[0].unsqueeze(0),
                                         hidden_states=input_hidden_state[0].unsqueeze(0),
                                         target_embeds=image_embeds[0].unsqueeze(0),
                                         save_path=image_save_path)

                    if self.cfg.log_validation > 0 and (self.train_step % self.cfg.log_validation == 10):
                        # Run validation
                        self.log_validation()

                    if self.train_step >= self.cfg.max_train_steps:
                        break

            self.train_dataloader, self.validation_dataloader = self.get_dataloaders()
        pbar.close()

    def preprocess_inputs(self, cond, text_encoder_concat=None):

        # Specific pre-processing for each mode
        if self.cfg.mode == 'instruct':
            input_image_embeds, input_hidden_state = pops_utils.preprocess(cond['source_a'], None,
                                                                    self.image_encoder,
                                                                    self.clip_mean.detach(),
                                                                    self.clip_std.detach(),
                                                                    concat_hidden_states=text_encoder_concat)
            # For CFG
            if random.random() < 0.2:
                input_hidden_state = torch.zeros_like(input_hidden_state)
                for b_ind in range(cond['source_a'].shape[0]):
                    cond['source_a'][b_ind] = cond['source_a'][b_ind] * 0
        elif self.cfg.mode == 'clothes':
            input_image_embeds, input_hidden_state = pops_utils.preprocess(None, None,
                                                                    self.image_encoder,
                                                                    self.clip_mean.detach(),
                                                                    self.clip_std.detach(),
                                                                    image_list=cond['crops'])
            if random.random() < 0.2:
                input_hidden_state = torch.zeros_like(input_hidden_state)
                # For vis purposes zero all images
                cond['crops'] = [torch.zeros_like(cond['crops'][0])]
        else:
            if self.cfg.mode == 'texture':
                should_drop_cond = []
                for b_ind in range(cond['source_a'].shape[0]):
                    # For CFG and random data
                    if 'is_random' in cond and cond['is_random'][b_ind]:
                        should_drop_a = random.random() < 0.5
                        should_drop_b = True
                    else:
                        # Randomly dropping one of the conditions to make the model use both?
                        should_drop_a = random.random() < 0.25
                        should_drop_b = random.random() < 0.25

                    if should_drop_a:
                        cond['source_a'][b_ind] = cond['source_a'][b_ind] * 0
                    if should_drop_b:
                        cond['source_b'][b_ind] = cond['source_b'][b_ind] * 0
                    should_drop_cond.append((should_drop_a, should_drop_b))
            elif self.cfg.mode == 'union':
                should_drop_cond = []
                for b_ind in range(cond['source_a'].shape[0]):
                    should_drop_a, should_drop_b = False, False
                    if ('is_random' in cond and cond['is_random'][b_ind]) or random.random() < 0.2:
                        should_drop_a = True
                        should_drop_b = True

                    if should_drop_a:
                        cond['source_a'][b_ind] = cond['source_a'][b_ind] * 0
                    if should_drop_b:
                        cond['source_b'][b_ind] = cond['source_b'][b_ind] * 0
                    should_drop_cond.append((should_drop_a, should_drop_b))
            elif self.cfg.mode == 'scene':
                should_drop_cond = []
                for b_ind in range(cond['source_a'].shape[0]):
                    should_drop_a, should_drop_b = False, False
                    if ('is_random' in cond and cond['is_random'][b_ind]) or (random.random() < 0.1):
                        should_drop_a = True
                        should_drop_b = True

                    if should_drop_a:
                        cond['source_a'][b_ind] = cond['source_a'][b_ind] * 0
                    if should_drop_b:
                        cond['source_b'][b_ind] = cond['source_b'][b_ind] * 0
                    should_drop_cond.append((should_drop_a, should_drop_b))
            else:
                raise ValueError(f"Invalid mode: {self.cfg.mode}")
            input_image_embeds, input_hidden_state = pops_utils.preprocess(cond['source_a'], cond['source_b'],
                                                                    self.image_encoder,
                                                                    self.clip_mean.detach(),
                                                                    self.clip_std.detach(),
                                                                    should_drop_cond=should_drop_cond)

        return input_image_embeds, input_hidden_state

    def log_validation(self):
        for sample_idx, batch in tqdm(enumerate(self.validation_dataloader)):
            image, cond = batch
            image = image.to(self.weight_dtype).to(self.accelerator.device)
            if 'source_a' in cond:
                cond['source_a'] = cond['source_a'].to(self.weight_dtype).to(self.accelerator.device)
            if 'source_b' in cond:
                cond['source_b'] = cond['source_b'].to(self.weight_dtype).to(self.accelerator.device)
            if 'crops' in cond:
                for crop_ind in range(len(cond['crops'])):
                    cond['crops'][crop_ind] = cond['crops'][crop_ind].to(self.weight_dtype).to(
                        self.accelerator.device)
            for key in cond.keys():
                if isinstance(cond[key], torch.Tensor):
                    cond[key] = cond[key].to(self.accelerator.device)

            text_mask = cond['mask']

            with torch.no_grad():
                target_embeds = self.image_encoder(image).image_embeds
                target_embeds = (target_embeds - self.clip_mean) / self.clip_std

                text_encoder_output = self.text_encoder(cond['input_ids'])
                text_encoder_hidden_states = text_encoder_output.last_hidden_state
                # Take only the unmasked tokens
                if self.cfg.mode == 'instruct':
                    text_encoder_concat = text_encoder_hidden_states[:, :text_mask.sum().item()]
                    input_image_embeds, input_hidden_state = pops_utils.preprocess(cond['source_a'],
                                                                            None,
                                                                            self.image_encoder,
                                                                            self.clip_mean.detach(),
                                                                            self.clip_std.detach(),
                                                                            concat_hidden_states=text_encoder_concat)
                elif self.cfg.mode == 'clothes':
                    input_image_embeds, input_hidden_state = pops_utils.preprocess(None, None,
                                                                            self.image_encoder,
                                                                            self.clip_mean.detach(),
                                                                            self.clip_std.detach(),
                                                                            image_list=cond['crops'])

                else:
                    input_image_embeds, input_hidden_state = pops_utils.preprocess(cond['source_a'],
                                                                            cond['source_b'],
                                                                            self.image_encoder,
                                                                            self.clip_mean.detach(),
                                                                            self.clip_std.detach())

            image_save_path = self.cfg.output_dir / 'val_images' / f"{self.train_step}_step_{sample_idx}_images.jpg"
            image_save_path.parent.mkdir(exist_ok=True, parents=True)

            save_target_image = image[0] if self.cfg.textures_dir is None else None
            conds_list = []
            if 'source_a' in cond:
                conds_list.append(cond['source_a'][0])
            if 'source_b' in cond:
                conds_list.append(cond['source_b'][0])
            if 'crops' in cond:
                for crop_ind in range(len(cond['crops'])):
                    conds_list.append(cond['crops'][crop_ind][0])
            # Apply the full diffusion process
            self.save_images(image=save_target_image, conds=conds_list,
                             label=cond['text_condition'][0] if 'text_condition' in cond else '',
                             input_embeds=input_image_embeds[0].unsqueeze(0),
                             hidden_states=input_hidden_state[0].unsqueeze(0),
                             target_embeds=target_embeds[0].unsqueeze(0),
                             save_path=image_save_path)

            if sample_idx == self.cfg.n_val_images:
                break
