#!/bin/bash

export CUDA_HOME=/usr/local/cuda-12.4

FIXED_SEED=42
BATCH_SIZE=1
MAX_SEQ_LEN=512
MAX_EPOCHS=30
VAL_CHECK_INTERVAL=30
LEARNING_RATE=5e-7
MODEL_NAME="./pretrained_lms/Qwen-Qwen3-8B"
PROMPT_UNCTRL="Here is a sentence {<INPUT_SRC>}; Here is its <TGT_LANG> translation {<OUTPUT_TGT>};"
PROMPT_CTRL="Here is a sentence {<INPUT_SRC>}; Here is its <TGT_LANG> translation written in <FORMALITY> style {<OUTPUT_TGT>};
The translated sentence conveys a <FORMALITY> style by using words such as <FORMALITY_TOKENS>."
PRETRAINED="<PRETRAINED_CHECKPOINT>"  # Ending in '_LoRA'
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

        echo "EXPERIMENT: ${EXPERIMENT_NAME} | Qwen3 (IT+CPO)"
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
          --max-epochs ${MAX_EPOCHS} \
          --batch-size ${BATCH_SIZE} \
          --padding-side "left" \
          --learning-rate ${LEARNING_RATE} \
          --fix-seed ${FIXED_SEED} \
          --strategy "ddp"  \
          --overwrite-arguments \
          --from-pretrained ${PRETRAINED} \
          --synthetic-pref-data ./data/CoCoA_MT/train/${SRC_LANG}-${TGT_LANG}/synthetic/preference_train_data/sonnet_3_ic_sampling_temp_ref_rej/synthetic_pref_data.csv \
          --only-synthetic

      fi

done
done
