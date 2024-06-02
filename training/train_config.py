from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union


@dataclass
class TrainConfig:
    # Dataset path
    dataset_path: Union[Path, List[Path]] = Path('datasets/generated/generated_things')
    # Validation dataset path
    val_dataset_path: Path = Path('datasets/generated/generated_things_val')
    # Path to pretrained model WITHOUT 2_1 folder
    cache_root: Path = Path('/tmp/kandinsky2')
    # The output directory where the model predictions and checkpoints will be written.
    output_dir: Path = Path('results/my_pops_model')
    # GPU device
    device: str = 'cuda:0'
    # The resolution for input images, all the images will be resized to this size
    img_size: int = 512
    # Batch size (per device) for the training dataloader
    train_batch_size: int = 1
    # Initial learning rate (after the potential warmup period) to use
    lr: float = 1e-5
    # Dataloader num workers.
    num_workers: int = 8
    # The beta1 parameter for the Adam optimizer.
    adam_beta1: float = 0.9
    # The beta2 parameter for the Adam optimizer
    adam_beta2: float = 0.999
    # Weight decay to use
    adam_weight_decay: float = 0.0  # 1e-2
    # Epsilon value for the Adam optimizer
    adam_epsilon: float = 1e-08
    # How often save images. Values less zero - disable saving
    log_image_frequency: int = 500
    # How often to run validation
    log_validation: int = 5000
    # The number of images to save during each validation
    n_val_images: int = 10
    # A seed for reproducible training
    seed: Optional[int] = None
    # The number of accumulation steps to use
    gradient_accumulation_steps: int = 1
    # Whether to use mixed precision training
    mixed_precision: Optional[str] = 'fp16'
    # Log to wandb
    report_to: str = 'wandb'
    # Path to pretrained prior model or model identifier from huggingface.co/models.
    pretrained_prior_path: str = 'kandinsky-community/kandinsky-2-2-prior'
    # Path to pretrained image encoder.
    pretrained_image_encoder: str = 'kandinsky-community/kandinsky-2-2-prior'
    # Path to scheduler.
    scheduler_path: str = 'kandinsky-community/kandinsky-2-2-prior'
    # Path to image_processor.
    image_processor_path: str = 'kandinsky-community/kandinsky-2-2-prior'
    # Path to text_encoder.
    text_encoder_path: str = 'kandinsky-community/kandinsky-2-2-prior'
    # Path to tokenizer.
    tokenizer_path: str = 'kandinsky-community/kandinsky-2-2-prior'
    use_ema: bool = False
    allow_tf32: bool = False
    use_8bit_adam: bool = False
    lr_scheduler: str = 'constant'
    # The number of training steps to run
    max_train_steps: int = 1000000
    # Max grad for clipping
    max_grad_norm: float = 1.0
    # How often to save checkpoints
    checkpointing_steps: int = 5000
    # The path to resume from
    resume_from_path: Optional[Path] = None
    # The step to resume from, mainly for logging
    resume_from_step: Optional[int] = None
    # Lora mode, untested
    lora_rank: Optional[int] = None
    # Which operator to train
    mode: str = 'texture'
    # The path to the textures dataset if used
    textures_dir: Optional[Path] = None
    # The path to the backgrounds dataset if used
    backgrounds_dir: Optional[Path] = None
    # optional directory of plain images to use for unconditional denoising
    randoms_dir: Optional[Path] = None
    # Whether full model is trained or only some layers, x_layers is the format for training only x layers
    training_mode: str = 'full'
    # Whether to use clip loss
    use_clip_loss: bool = False
    # Clip lambda
    clip_strength: float = 10.0
