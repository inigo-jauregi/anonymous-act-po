#!/bin/bash

export CUDA_HOME=/usr/local/cuda-12.4

FIXED_SEED=42
BATCH_SIZE=4
MAX_SEQ_LEN=512
MAX_EPOCHS=30
MODEL_NAME="facebook/nllb-200-distilled-600M"
PROMPT_UNCTRL="<INPUT_SRC>"
PROMPT_CTRL="<FORMALITY> <INPUT_SRC>"

for SRC_LANG in "en"; do
  for TGT_LANG in "de"; do

      TRAIN_PATH="./data/CoCoA_MT/train/${SRC_LANG}-${TGT_LANG}/formality-control.train90.all"
      VAL_PATH="./data/CoCoA_MT/train/${SRC_LANG}-${TGT_LANG}/formality-control.val10.all"
      TEST_PATH="./data/CoCoA_MT/test/${SRC_LANG}-${TGT_LANG}/formality-control.test.all"

      EXPERIMENT_NAME="PAPER_cocoa_mt_${SRC_LANG}-${TGT_LANG}"

      # zero-shot un-controlled experiments
      if [[ "$SRC_LANG" != "${TGT_LANG}" ]]; then

        echo "EXPERIMENT: ${EXPERIMENT_NAME} | NLLB (ZS)"
        python3 -m scripts.cocoa_mt_experiments.train_model \
            --model-name ${MODEL_NAME} \
            --train-data ${TRAIN_PATH} \
            --val-data ${VAL_PATH} \
            --test-data ${TEST_PATH} \
            --experiment-name ${EXPERIMENT_NAME} \
            --src-lng ${SRC_LANG} \
            --tgt-lng ${TGT_LANG} \
            --just-test \
            --prompt "${PROMPT_UNCTRL}" \
            --max-length ${MAX_SEQ_LEN} \
            --batch-size ${BATCH_SIZE} \
            --fix-seed ${FIXED_SEED}


        echo "EXPERIMENT: ${EXPERIMENT_NAME} | NLLB (IT)"
        python3 -m scripts.cocoa_mt_experiments.train_model \
            --model-name ${MODEL_NAME} \
            --train-data ${TRAIN_PATH} \
            --val-data ${VAL_PATH} \
            --test-data ${TEST_PATH} \
            --experiment-name ${EXPERIMENT_NAME} \
            --src-lng ${SRC_LANG} \
            --tgt-lng ${TGT_LANG} \
            --prompt "${PROMPT_CTRL}" \
            --max-length ${MAX_SEQ_LEN} \
            --log-model-mlflow \
            --max-epochs ${MAX_EPOCHS} \
            --fix-seed ${FIXED_SEED} \
            --strategy "ddp"  # Needed for multi-GPU training and validation monitoring
      fi

done
done
