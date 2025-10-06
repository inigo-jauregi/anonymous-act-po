#!/bin/bash

export CUDA_HOME=/usr/local/cuda-12.4

FIXED_SEED=42
BATCH_SIZE=4
MAX_SEQ_LEN=512
MAX_EPOCHS=30
MODEL_NAME="./pretrained_lms/Qwen-Qwen3-8B"
PROMPT_UNCTRL="Here is a sentence {<INPUT_SRC>}; Here is its <TGT_LANG> translation {<OUTPUT_TGT>};"
PROMPT_CTRL="Here is a sentence {<INPUT_SRC>}; Here is its <TGT_LANG> translation in which every mentioned person's gender is <GENDER> {<OUTPUT_TGT>};
In the translation, the <GENDER> gender of the person is made explicit by words such as <GENDER_TOKENS>."

for SRC_LANG in "en"; do
  for TGT_LANG in "de"; do

      TRAIN_PATH="./data/MT_GenEval/dataset_uts/${SRC_LANG}-${TGT_LANG}/train"
      VAL_PATH="./data/MT_GenEval/dataset_uts/${SRC_LANG}-${TGT_LANG}/val"
      TEST_PATH="./data/MT_GenEval/dataset_uts/${SRC_LANG}-${TGT_LANG}/test"

      EXPERIMENT_NAME="PAPER_mt_geneval_${SRC_LANG}-${TGT_LANG}"

      # zero-shot un-controlled experiments
      if [[ "$SRC_LANG" != "${TGT_LANG}" ]]; then

        echo "EXPERIMENT: ${EXPERIMENT_NAME} | Qwen3 (ZS)"
        python3 -m scripts.mt_geneval_experiments.train_model \
            --model-name ${MODEL_NAME} \
            --train-data ${TRAIN_PATH} \
            --val-data ${VAL_PATH} \
            --test-data ${TEST_PATH} \
            --experiment-name ${EXPERIMENT_NAME} \
            --src-lng ${SRC_LANG} \
            --tgt-lng ${TGT_LANG} \
            --just-test \
            --padding-side "left" \
            --prompt "${PROMPT_CTRL}" \
            --max-length ${MAX_SEQ_LEN} \
            --batch-size ${BATCH_SIZE} \
            --fix-seed ${FIXED_SEED}

        echo "EXPERIMENT: ${EXPERIMENT_NAME} | Qwen3-LoRA (IT)"
        python3 -m scripts.mt_geneval_experiments.train_model \
            --model-name ${MODEL_NAME} \
            --train-data ${TRAIN_PATH} \
            --val-data ${VAL_PATH} \
            --test-data ${TEST_PATH} \
            --experiment-name ${EXPERIMENT_NAME} \
            --src-lng ${SRC_LANG} \
            --tgt-lng ${TGT_LANG} \
            --prompt "${PROMPT_CTRL}" \
            --max-length ${MAX_SEQ_LEN} \
            --padding-side "left" \
            --lora-finetuning \
            --log-model-mlflow \
            --max-epochs ${MAX_EPOCHS} \
            --fix-seed ${FIXED_SEED} \
            --patience 3 \
            --strategy "ddp"
      fi

done
done
