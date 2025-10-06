#!/bin/bash

export CUDA_HOME=/usr/local/cuda-12.4
export AWS_PROFILE=<AWS_PROFILE>


MODEL_SONNET="anthropic.claude-3-sonnet-20240229-v1:0"
# prompt_claude = "Here is a sentence {<INPUT_SRC>}; Please provide the <TGT_LANG> translation between {}: {<OUTPUT_TGT>};"
PROMPT="Here is a sentence {<INPUT_SRC>}; Please provide the <TGT_LANG> translation written in <FORMALITY> style between curly brackets: {<OUTPUT_TGT>};
        The translated sentence conveys a formal style by using words such as <FORMALITY_TOKENS>."

for SRC_LANG in "en"; do
  for TGT_LANG in "de"; do
    for TEMP in 0.0 0.2 0.4 0.5 0.6 0.8 0.9 1.0; do
      for ITER in 1 2 3 4; do

        TRAIN_PATH="./data/CoCoA_MT/train/${SRC_LANG}-${TGT_LANG}/formality-control.train90.all_BOTH"
        VAL_PATH="./data/CoCoA_MT/train/${SRC_LANG}-${TGT_LANG}/formality-control.val10.all_BOTH"
        TEST_PATH="./data/CoCoA_MT/train/${SRC_LANG}-${TGT_LANG}/formality-control.train90.all_BOTH"

        # zero-shot un-controlled experiments
        if [[ "$SRC_LANG" != "${TGT_LANG}" ]]; then
          EXPERIMENT_NAME="synthetic_cocoa_sonnet_3_ic_${SRC_LANG}-${TGT_LANG}"
          echo "SAMPLE GENERATION for ${SRC_LANG} to ${TGT_LANG} - Iter ${ITER}"
          python3 -m scripts.cocoa_mt_experiments.train_model \
              --model-name ${MODEL_SONNET} \
              --experiment-name ${EXPERIMENT_NAME} \
              --train-data ${TRAIN_PATH} \
              --val-data ${VAL_PATH} \
              --test-data ${TEST_PATH} \
              --src-lng ${SRC_LANG} \
              --tgt-lng ${TGT_LANG} \
              --just-test \
              --prompt "${PROMPT}" \
              --s3-bucket "<S3_bucket>" \
              --iam-role "<IAM_ROLE>" \
              --max-length 2048 \
              --in-context-learning \
              --in-context-num-samples 8 \
              --temperature ${TEMP} \
              --devices '[1]'
        fi
done
done
done
done