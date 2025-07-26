# FLUX.1 Kontext LoRA Trainer for Style Transfer

> [!NOTE]
> This training framework was used to create the **20+ style LoRA models** available on the Hugging Face Hub at [Owen777/Kontext-Style-Loras](https://huggingface.co/Owen777/Kontext-Style-Loras). This repository contains the exact code to replicate those models and train your own.

This repository provides a training script to fine-tune the `FLUX.1-Kontext-dev` model using LoRA (Low-Rank Adaptation) for style transfer tasks. It is designed to work with paired image-to-image datasets, where each sample consists of an original image and its stylized counterpart.

The training process is powered by Hugging Face's `diffusers`, `accelerate`, and `peft` libraries for efficient and scalable training on multi-GPU setups.

## Features

- **LoRA Fine-tuning**: Efficiently adapts the powerful `FluxTransformer2DModel` for new styles without training the entire model.
- **Style-Specific Training**: Train on one or more specific artistic styles using a structured dataset format.
- **Efficient Training**: Leverages `accelerate` for multi-GPU distributed training and `bitsandbytes` for 8-bit Adam optimization to reduce memory footprint.
- **Mixed-Precision**: Supports `fp16` and `bf16` mixed-precision training to speed up computations.
- **Customizable**: Easily configure LoRA rank, alpha, learning rate, and other hyperparameters.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Kontext-Lora-Trainer.git
    cd Kontext-Lora-Trainer
    ```


2.  **Download Base Model:**
    The training script requires a local checkpoint of the base model. You can download the official `FLUX.1-Kontext-dev` model from the Hugging Face Hub or use your own compatible checkpoint.
    The `run.sh` script assumes it is located in a directory named `checkpoint_kontext`.

## Dataset Preparation

The training script uses a custom `OmniConsistencyDataset` loader that expects a specific directory structure.

- The root data directory (e.g., `./dataset/OmniConsistency`) should contain subdirectories, each named after a specific style (e.g., `Paper_Cutting`, `Van_Gogh`).
- Inside each style directory, there must be a `train.jsonl` file.
- Each line in the `.jsonl` file represents a training sample and should be a JSON object containing the following keys:
    - `src`: Relative path to the source (original) image.
    - `tar`: Relative path to the target (stylized) image.
    - `prompt`: A text prompt (although the current script can generate prompts based on style type).

**Example Directory Structure:**

```
./dataset/OmniConsistency/
├── Paper_Cutting/
│   ├── images/
│   │   ├── 001_src.png
│   │   └── 001_tar.png
│   └── train.jsonl
│
└── Van_Gogh/
    ├── images/
    │   ├── 001_src.png
    │   └── 001_tar.png
    └── train.jsonl
```

**Note:** In the current `train.py` script, the dataset path is hardcoded as `root_dir="./dataset/OmniConsistency"`. You may want to modify the script to accept a command-line argument for the dataset path for better flexibility.

## Training

The `accelerate` library is used to launch the training script. You can configure `accelerate` for your specific hardware setup by running `accelerate config`.

The `run.sh` script provides a template for launching a training job. Below is a documented version of the command:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

accelerate launch train.py \
  --pretrained_model_name_or_path="checkpoint_kontext" \
  --output_dir="kontext-style-lora-Paper_Cutting" \
  --style_type="Paper_Cutting" \
  --mixed_precision="bf16" \
  --resolution=1024 \
  --train_batch_size=8 \
  --gradient_accumulation_steps=1 \
  --guidance_scale=1.0 \
  --rank=256 \
  --lora_alpha=256 \
  --learning_rate=5e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --optimizer="adamw" \
  --use_8bit_adam \
  --gradient_checkpointing \
  --checkpointing_steps=500 \
  --validation_prompt="A photo of a cat in the paper cutting style" \
  --seed=42
```

### Key Training Arguments

- `--pretrained_model_name_or_path`: Path to the local base model checkpoint.
- `--output_dir`: Directory to save the trained LoRA weights and checkpoints.
- `--style_type`: The name of the style to train on. This must match one of the subdirectories in your dataset folder.
- `--mixed_precision`: Use `bf16` or `fp16` for faster training.
- `--resolution`: The training image resolution. Should match the base model's optimal resolution (1024 for FLUX.1).
- `--train_batch_size`: Batch size per GPU.
- `--rank` & `--lora_alpha`: The rank and alpha for the LoRA layers. A higher rank allows for more expressive power at the cost of a larger model size.
- `--learning_rate`: The initial learning rate.
- `--max_train_steps`: Total number of training steps.
- `--validation_prompt`: A prompt used to generate sample images for validation.
- `--checkpointing_steps`: Save a full training state every N steps.

## Using the Trained LoRA

After training, the final LoRA weights will be saved in your specified `--output_dir` as `pytorch_lora_weights.safetensors`. You can load these weights into a `diffusers` pipeline for inference.

```python
from diffusers import FluxKontextPipeline
import torch

# Load the base pipeline
pipeline = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev", 
    torch_dtype=torch.bfloat16
).to('cuda')

# Load your trained LoRA weights
lora_path = "kontext-style-lora-Paper_Cutting/pytorch_lora_weights.safetensors"
pipeline.load_lora_weights(lora_path)

# Prepare a condition image and prompt
condition_image = ... # Load your source image with PIL
prompt = "A beautiful castle, in the paper cutting style"

# Run inference
image = pipeline(prompt=prompt, image=condition_image).images[0]
image.save("stylized_image.png")
```

## Acknowledgements
- This work is built upon the [FLUX.1](https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux) model by Black Forest Labs.
- Heavily utilizes the [Hugging Face diffusers](https://github.com/huggingface/diffusers) library.