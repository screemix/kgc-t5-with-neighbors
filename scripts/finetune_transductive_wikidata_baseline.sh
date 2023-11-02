#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./test_bert_sparse_pretrain_train_valid.sh
set -e
cd ..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_NAME=t5-small
TOKENIZER_NAME=tokenizer-t5-wikidata5m
MODEL_TYPE=encoder-decoder
TASK_NAME=not-pretrained-wikidata-baseline-from-checkpoint

TGT_LEN=512

METRIC=exact_match
SCHEDULER=constant_with_warmup
ITERS=2000000
TBS=320
BS=32
MODEL_CFG="t5-small"

for SRC_LEN in 512
do
for LR in 1e-06
do
for N in 1
do
echo $MODEL_CFG
horovodrun --gloo -np $NP python run_finetuning_kglm.py \
        --task_name $TASK_NAME \
        --model_path ./runs/$MODEL_NAME/$TASK_NAME/lr${LR}_${SCHEDULER}_adamw_wd1e-03_${SRC_LEN}-${TGT_LEN}_bs${TBS}_iters${ITERS}_pretrained/run_$N \
        --model_cfg $MODEL_CFG \
        --tokenizer $TOKENIZER_NAME \
        --model_type $MODEL_TYPE \
        --model_cls transformers:T5ForConditionalGeneration \
        --use_generate_on_valid \
        --save_best \
        --drop_neighborhood \
        --input_seq_len $SRC_LEN \
        --target_seq_len $TGT_LEN \
        --batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
        --iters $ITERS \
        --optimizer AdamW  --weight_decay 0.001 \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps $(($ITERS/10)) \
        --data_n_workers 2 \
        --log_interval $(($ITERS/2000)) --valid_interval $(($ITERS/200)) \
        --show_valid_examples 10 \
        --optimize_metric $METRIC --optimize_mode max \
        --seed $(($N+42))
done
done
done
done
echo "run_pretraining.py done"
echo "done"
