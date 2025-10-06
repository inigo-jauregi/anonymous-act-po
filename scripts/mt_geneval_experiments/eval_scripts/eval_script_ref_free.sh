#!/bin/bash

export CUDA_HOME=/usr/local/cuda-12.4

COMET_MODEL="Unbabel/wmt22-comet-da"
REF_FREE_COMET_MODEL="./pretrained_lms/Unbabel-wmt23-cometkiwi-da-xxl/checkpoints/model.ckpt"

for SRC_LANG in "en"; do
  for TGT_LANG in "de"; do

    echo "Evaluating ${SRC_LANG}-${TGT_LANG}"

    python -m scripts.general.pred_eval \
                --preds-file "PAPER_mt_geneval_${SRC_LANG}-${TGT_LANG}" \
                --src-lng ${SRC_LANG} \
                --tgt-lng ${TGT_LANG} \
                --include-ref-free-comet ${REF_FREE_COMET_MODEL} \
                --include-comet ${COMET_MODEL} \
                --only-eval-true
done
done
