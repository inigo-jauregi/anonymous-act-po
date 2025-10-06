#!/bin/bash

export CUDA_HOME=/usr/local/cuda-12.4

COMET_MODEL="Unbabel/wmt22-comet-da"
REF_FREE_COMET_MODEL="./pretrained_lms/Unbabel-wmt23-cometkiwi-da-xxl/checkpoints/model.ckpt"

SRC_LANG=("en")
TGT_LANG=("de")

for i in "${!SRC_LANG[@]}"; do

  echo "Evaluating ${SRC_LANG[i]}-${TGT_LANG[i]}"

  # zero-shot un-controlled experiments
  if [[ "${SRC_LANG[i]}" != "${TGT_LANG}" ]]; then

    python -m scripts.general.pred_eval \
                --preds-file "PAPER_fame_mt_${SRC_LANG[i]}-${TGT_LANG[i]}" \
                --src-lng ${SRC_LANG[i]} \
                --tgt-lng ${TGT_LANG[i]} \
                --include-ref-free-comet ${REF_FREE_COMET_MODEL} \
                --include-comet ${COMET_MODEL} \
                --only-eval-true \
#                --fix-cocoa-preds ./data/CoCoA_MT/test/${SRC_LANG}-${TGT_LANG}/formality-control.test.all \
#                --gpt-judge \

  fi

done
