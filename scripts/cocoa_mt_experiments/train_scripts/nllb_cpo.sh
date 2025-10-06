#!/bin/bash

export CUDA_HOME=/usr/local/cuda-12.4

FIXED_SEED=42
BATCH_SIZE=1
MAX_SEQ_LEN=512
LEARNING_RATE=5e-6
MAX_EPOCHS=30
VAL_CHECK_INTERVAL=30
MODEL_NAME="facebook/nllb-200-distilled-600M"
PROMPT_UNCTRL="<INPUT_SRC>"
PROMPT_CTRL="<FORMALITY> <INPUT_SRC>"
PRETRAINED="<PRETRAINED_CHECKPOINT>"  # Ending in .ckpt
BETA=1
LAMBDA=0.25

for SRC_LANG in "en"; do
  for TGT_LANG in "de"; do

      TRAIN_PATH="./data/CoCoA_MT/train/${SRC_LANG}-${TGT_LANG}/formality-control.train90.all"
      VAL_PATH="./data/CoCoA_MT/train/${SRC_LANG}-${TGT_LANG}/formality-control.val10.all"
      TEST_PATH="./data/CoCoA_MT/test/${SRC_LANG}-${TGT_LANG}/formality-control.test.all"

      EXPERIMENT_NAME="PAPER_cocoa_mt_${SRC_LANG}-${TGT_LANG}"

      # zero-shot un-controlled experiments
      if [[ "$SRC_LANG" != "${TGT_LANG}" ]]; then

        echo "EXPERIMENT: ${EXPERIMENT_NAME} | NLLB (IT+CPO)"
        python3 -m scripts.cocoa_mt_experiments.train_model \
          --model-name ${MODEL_NAME} \
          --train-data ${TRAIN_PATH} \
          --val-data ${VAL_PATH} \
          --test-data ${TEST_PATH} \
          --experiment-name ${EXPERIMENT_NAME} \
          --src-lng ${SRC_LANG} \
          --tgt-lng ${TGT_LANG} \
          --prompt "${PROMPT_CTRL}" \
          --objective cpo \
          --dpo-beta ${BETA} \
          --cpo-lambda ${LAMBDA} \
          --val-check-interval ${VAL_CHECK_INTERVAL} \
          --max-length ${MAX_SEQ_LEN} \
          --batch-size ${BATCH_SIZE} \
          --learning-rate ${LEARNING_RATE} \
          --fix-seed ${FIXED_SEED} \
          --strategy "ddp"  \
          --overwrite-arguments \
          --from-pretrained ${PRETRAINED}


      fi

done
done
