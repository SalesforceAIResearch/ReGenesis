#!/bin/bash
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}$
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# change
mistral_path="ur/path/to/Mistral-7B-Instruct-v0.3"
model_short_name="mistral-v3"
data_name=strategyqa_em_box_solution_max3_mv_sd_4_FT_ablation
num_train_epochs=3
train_data_path="ur-path-to/FT-data.json"

model_name=${data_name}_${model_short_name}_e${num_train_epochs}_1e6
output_dir="/output-path/${model_name}"

torchrun --nproc_per_node=8 ./finetune_code.py \
    --model_name_or_path ${mistral_path} \
    --data_path ${train_data_path} \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 20000000000 \
    --save_total_limit 1 \
    --learning_rate 1e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 2000 \
    --fsdp "full_shard offload auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'MistralDecoderLayer' \
    --tf32 False \
    --gradient_checkpointing True \
    --run_name ${model_name} \
    --report_to wandb