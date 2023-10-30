#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./test_bert_sparse_pretrain_train_valid.sh
set -e
cd ..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_NAME=t5-small
MODEL_CFG="t5-small"
TOKENIZER_NAME=tokenizer-t5-wikidata5m
MODEL_TYPE=encoder-decoder
TASK_NAME=not-pretrained-wikidata

TGT_LEN=512

METRIC=exact_match
ITERS=150000
TBS=320
BS=40
N=1
SRC_LEN=512
NUM_TRAINING_STEPS=500000


for LR in 1e-04 2e-04 5e-04       
do
    for SCHEDULER in constant_with_warmup linear cosine
    do

        for NUM_WARMUP_STEPS in 10000 20000 50000
        do

            echo $MODEL_CFG
            echo $LR
            echo $SCHEDULER
            echo $NUM_WARMUP_STEPS
            echo $NUM_TRAINING_STEPS
            horovodrun --gloo -np $NP python run_finetuning_kglm.py \
                    --task_name $TASK_NAME \
                    --model_path ./runs/$MODEL_NAME/$TASK_NAME/lr${LR}_${SCHEDULER}_steps_${NUM_WARMUP_STEPS}_with_training_steps_${NUM_TRAINING_STEPS}_adamw_wd1e-03_${SRC_LEN}-${TGT_LEN}_bs${TBS}_iters${ITERS}_pretrained/run_$N \
                    --model_cfg $MODEL_CFG \
                    --tokenizer $TOKENIZER_NAME \
                    --model_type $MODEL_TYPE \
                    --model_cls transformers:T5ForConditionalGeneration \
                    --use_generate_on_valid \
                    --save_best \
                    --input_seq_len $SRC_LEN \
                    --target_seq_len $TGT_LEN \
                    --batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
                    --iters $ITERS \
                    --optimizer AdamW  --weight_decay 0.001 \
                    --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps $(($NUM_WARMUP_STEPS)) --num_training_steps $NUM_TRAINING_STEPS \
                    --data_n_workers 2 \
                    --log_interval $(($ITERS/2000)) --valid_interval $(($ITERS/100)) \
                    --show_valid_examples 10 \
                    --optimize_metric $METRIC --optimize_mode max \
                    --seed $(($N+42))
        done
    done
done


echo "run_pretraining.py done"
echo "done"
