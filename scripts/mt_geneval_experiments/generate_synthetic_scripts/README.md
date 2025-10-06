# Create synthetic data from samples

This folder contains scripts to generate the
synthetic contrastive data from generated samples.


See example below (from the `claude_synthetic_prefs.sh` script:

```bash
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
```

**Notes**
- Run the script from the parent directory
- Change the Mlflow experiment name to match the experiment name accordingly.
- Change the `src` and `tgt` languages to match the language pair from the experiment
- This script will generate synthetic data forcing the reference to be the rejected sample.