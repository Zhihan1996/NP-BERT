export GLUE_DIR=/home/share/data/GLUE/
export CUDA_VISIBLE_DEVICES="3" 
export TASK_NAME=MRPC

python ./run_glue.py \
    --config_name "/home/share/pytorch-transformers/config.json" \
    --model_type bert \
    --model_name_or_path "/home/share/pytorch-transformers/wwm_uncased_L-24_H-1024_A-16" \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=1   \
    --per_gpu_train_batch_size=1   \
    --gradient_accumulation_steps=32 \
    --learning_rate 2e-5 \
    --num_train_epochs 5.0 \
    --save_steps 2000 \
    --output_dir /home/share/finetuned-models/bert-large-uncased-wwm/$TASK_NAME/gcs_32_lr2-5_ep4_freezeemb_new/ \
    --overwrite_output_dir
