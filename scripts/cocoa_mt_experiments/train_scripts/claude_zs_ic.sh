#!/bin/bash

export CUDA_HOME=/usr/local/cuda-12.4

FIXED_SEED=42
MAX_SEQ_LEN=2048
MODEL_SONNET="anthropic.claude-3-sonnet-20240229-v1:0"
PROMPT_UNCTRL="Here is a sentence {<INPUT_SRC>}; Please provide the <TGT_LANG> translation between {}: {<OUTPUT_TGT>};"
PROMPT_CTRL="Here is a sentence {<INPUT_SRC>}; Please provide the <TGT_LANG> translation written in <FORMALITY> style between curly brackets: {<OUTPUT_TGT>};
        The translated sentence conveys a <FORMALITY> style by using words such as <FORMALITY_TOKENS>."

  for SRC_LANG in "hi"; do
    for TGT_LANG in "es"; do

      TRAIN_PATH="./data/CoCoA_MT/train/${SRC_LANG}-${TGT_LANG}/formality-control.train90.all"
      VAL_PATH="./data/CoCoA_MT/train/${SRC_LANG}-${TGT_LANG}/formality-control.val10.all"
      TEST_PATH="./data/CoCoA_MT/test/${SRC_LANG}-${TGT_LANG}/formality-control.test.all"

      EXPERIMENT_NAME="PAPER_cocoa_mt_${SRC_LANG}-${TGT_LANG}"

      # zero-shot un-controlled experiments
      if [[ "$SRC_LANG" != "${TGT_LANG}" ]]; then

        echo "EXPERIMENT: ${EXPERIMENT_NAME} | Claude 3 (ZS) | ${MODEL_NAME}"
        python3 -m scripts.cocoa_mt_experiments.train_model \
            --model-name ${MODEL_NAME} \
            --experiment-name ${EXPERIMENT_NAME} \
            --train-data ${TRAIN_PATH} \
            --val-data ${VAL_PATH} \
            --test-data ${TEST_PATH} \
            --src-lng ${SRC_LANG} \
            --tgt-lng ${TGT_LANG} \
            --just-test \
            --prompt "${PROMPT_CTRL}" \
            --s3-bucket "<S3_bucket>" \
            --iam-role "<IAM_ROLE>" \
            --max-length ${MAX_SEQ_LEN} \
            --temperature 0 \
            --fix-seed ${FIXED_SEED}

        # In context learning experiments
        for NUM_SHOTS in 1 4 8; do
          echo "EXPERIMENT: ${EXPERIMENT_NAME} | Claude 3 (IC ${NUM_SHOTS}-SHOTS) | ${MODEL_NAME}"
          python3 -m scripts.cocoa_mt_experiments.train_model \
              --model-name ${MODEL_NAME} \
              --train-data ${TRAIN_PATH} \
              --val-data ${VAL_PATH} \
              --test-data ${TEST_PATH} \
              --experiment-name ${EXPERIMENT_NAME} \
              --src-lng ${SRC_LANG} \
              --tgt-lng ${TGT_LANG} \
              --just-test \
              --prompt "${PROMPT_CTRL}" \
              --s3-bucket "<S3_bucket>" \
              --iam-role "<IAM_ROLE>" \
              --max-length ${MAX_SEQ_LEN} \
              --temperature 0.0 \
              --fix-seed ${FIXED_SEED} \
              --in-context-learning \
              --in-context-num-samples ${NUM_SHOTS}
        done
      fi

  done
done

