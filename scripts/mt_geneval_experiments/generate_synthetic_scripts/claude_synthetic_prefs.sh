#!/bin/bash

export CUDA_HOME=/usr/local/cuda-12.4
export AWS_PROFILE=<AWS_PROFILE>

EXPERIMENT_NAME="synthetic_geneval_sonnet_3_en-de"
SRC_LANG="en"
TGT_LANG="de"

echo "Generate preference data for ${SRC_LANG} to ${TGT_LANG}"
python3 -m scripts.mt_geneval_experiments.generate_synthetic_data \
    --experiment-name ${EXPERIMENT_NAME} \
    --src ${SRC_LANG} \
    --tgt ${TGT_LANG} \
    --fixed 'rej'
