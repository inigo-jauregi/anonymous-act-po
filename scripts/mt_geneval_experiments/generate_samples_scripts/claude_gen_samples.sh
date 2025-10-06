#!/bin/bash

export CUDA_HOME=/usr/local/cuda-12.4
export AWS_PROFILE=<AWS_PROFILE>

FIXED_SEED=42
MAX_SEQ_LEN=2048
MODEL_SONNET="anthropic.claude-3-sonnet-20240229-v1:0"
PROMPT_UNCTRL="Here is a sentence {<INPUT_SRC>}; Please provide the <TGT_LANG> translation between {}: {<OUTPUT_TGT>};"
PROMPT_CTRL="Here is a sentence {<INPUT_SRC>}; Please provide the <TGT_LANG> translation in which every mentioned person's gender is <GENDER> between curly brackets: {<OUTPUT_TGT>};
        In the translation, the <GENDER> gender of the person is made explicit by words such as <GENDER_TOKENS>."

for SRC_LANG in "en"; do
  for TGT_LANG in "de"; do
    for TEMP in 0.0 0.2 0.4 0.5 0.6 0.8 0.9 1.0; do
      for ITER in 1 2 3 4; do

        TRAIN_PATH="./data/MT_GenEval/dataset_uts/${SRC_LANG}-${TGT_LANG}/train"
        VAL_PATH="./data/MT_GenEval/dataset_uts/${SRC_LANG}-${TGT_LANG}/val"
        TEST_PATH="./data/MT_GenEval/dataset_uts/${SRC_LANG}-${TGT_LANG}/train"

        # zero-shot un-controlled experiments
        if [[ "$SRC_LANG" != "${TGT_LANG}" ]]; then
          EXPERIMENT_NAME="synthetic_mt_geneval_sonnet_3_ic_${SRC_LANG}-${TGT_LANG}"
          echo "SAMPLE GENERATION for ${SRC_LANG} to ${TGT_LANG} - Iter ${ITER}"
          python3 -m scripts.mt_geneval_experiments.train_model \
              --model-name ${MODEL_SONNET} \
              --experiment-name ${EXPERIMENT_NAME} \
              --train-data ${TRAIN_PATH} \
              --val-data ${VAL_PATH} \
              --test-data ${TEST_PATH} \
              --src-lng ${SRC_LANG} \
              --tgt-lng ${TGT_LANG} \
              --just-test \
              --prompt "${PROMPT_CTRL}" \
              --s3-bucket "<S3_BUCKET>" \
              --iam-role "<IAM_ROLE>" \
              --max-length 2048 \
              --in-context-learning \
              --in-context-num-samples 8 \
              --temperature ${TEMP}
        fi
done
done
done
done