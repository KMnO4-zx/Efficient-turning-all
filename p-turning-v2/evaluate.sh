PRE_SEQ_LEN=128
CHECKPOINT=adgen-chatglm2-6b-pt-128-2e-2
STEP=3000
NUM_GPUS=1

CUDA_VISIBLE_DEVICES=0 python main.py \
    --do_predict \
    --validation_file paper_keywords_test_p2_b.json \
    --test_file paper_keywords_test_p2_b.json \
    --overwrite_cache \
    --prompt_column content \
    --response_column summary \
    --model_name_or_path ../../../data-1/ \
    --ptuning_checkpoint ./output/$CHECKPOINT/checkpoint-$STEP \
    --output_dir ./output/$CHECKPOINT \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_eval_batch_size 16 \
    --predict_with_generate \
    --pre_seq_len $PRE_SEQ_LEN \
