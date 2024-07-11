DATA_PATH="/kaggle/working/svgtopng_dataset/custom-dataset.json"
IMAGE_PATH="/kaggle/working/svgtopng_dataset/images"
MODEL_MAX_LENGTH=3072
OUTPUT_DIR="/kaggle/working/custom-finetune-TinyLLaVA-Phi-2-SigLIP-3.1B-lora"

deepspeed --include /device:TPU:0,/device:TPU:1,/device:TPU:2,/device:TPU:3,/device:TPU:4,/device:TPU:5,/device:TPU:6,/device:TPU:7 --master_port 29501 /kaggle/working/TinyLLaVA_Factory/tinyllava/train/custom_finetune.py \
    --deepspeed /kaggle/working/TinyLLaVA_Factory/scripts/zero2.json \
    --data_path  $DATA_PATH \
    --image_folder $IMAGE_PATH \
    --is_multimodal True \
    --conv_version phi \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio square \
    --training_recipe lora \
    --tune_type_llm lora \
    --tune_type_vision_tower frozen \
    --tune_vision_tower_from_layer 0 \
    --tune_type_connector full \
    --lora_r 128 \
    --lora_alpha 256 \
    --group_by_modality_length False \
    --pretrained_model_path "tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B" \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --tokenizer_use_fast False \
    --run_name custom-finetune-TinyLLaVA-Phi-2-SigLIP-3.1B-lora
