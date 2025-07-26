#!/bin/bash
# export HF_HOME=$(pwd)/cache
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

accelerate launch train.py \
  --pretrained_model_name_or_path=checkpoint_kontext  \
  --output_dir="kontext-style-lora-ALL" \
  --mixed_precision="bf16" \
  --resolution=1024 \
  --train_batch_size=8 \
  --guidance_scale=1.0 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --optimizer="adamw" \
  --style_type="ALL" \
  --rank=256 \
  --lora_alpha=256 \
  --use_8bit_adam \
  --learning_rate=5e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --checkpointing_steps=500 \
  --validation_epochs=50000 \
  --validation_prompt="Turn this image into the paper cutting style" \
  --num_validation_images=2 \
  --dataloader_num_workers=4 \
  --max_sequence_length=512 \
  --weighting_scheme="none" \
  --seed=42 






#"3D_Chibi" "American_Cartoon" "Chinese_Ink" "Clay_Toy" "Fabric" "Ghibli" "Irasutoya"