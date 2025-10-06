import os
import ast
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"
import shutil
import argparse

import mlflow
from pytorch_lightning import seed_everything

from src.ctrlpost.models import create_model_from_name
from src.ctrlpost.dataloaders import MtGenEvalDataModule
from src.ctrlpost.trainer import create_trainer
from src.ctrlpost.utils import custom_load_from_checkpoint


# Example usage
def main(args):

    # Set seed for all random number generators
    if args.fix_seed:
        seed_everything(args.fix_seed, workers=True)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
    checkpoint_dir = f'./checkpoints_{timestamp}'
    # pretraiend_model_hf = 'facebook/nllb-200-distilled-600M'
    # prompt = "<FORMALITY> <INPUT_SRC>"
    model = create_model_from_name(**vars(args))

    lora_finetuning = args.lora_finetuning
    # If loading from pretrained checkpoint, load checkpoint from MLFlow
    if args.from_pretrained:
        # extract mlflow path
        pretrained_path_ckpt = mlflow.artifacts.download_artifacts(artifact_uri=args.from_pretrained)
        model = custom_load_from_checkpoint(model, pretrained_path_ckpt, args, strict=False)
        if hasattr(model, "overwrite_ref_model_parameters"):
            if model.flag_overwrite_ref_model_parameters:
                model.overwrite_ref_model_parameters()
        # If the checkpoint path ends in LoRA, then overwrite the lora_finetuning argument, in order to only save the adapter
        # as the checkpoint
        if pretrained_path_ckpt.endswith('_LoRA'):
            lora_finetuning = True

    # Add variable to inform the model is controllable or not
    if '<GENDER>' in args.prompt:
        model.controllable = True
    else:
        model.controllable = False

    # Create data module
    datamodule = MtGenEvalDataModule(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        test_data_path=args.test_data,
        src_lang=model.src_lng,
        tgt_lang=model.tgt_lng,
        tokenizer=model.tokenizer,
        model_type=model.model_type,
        batch_size=args.batch_size,
        prompt=args.prompt,
        max_length=args.max_length,
        in_context_learning=args.in_context_learning,
        in_context_num_samples=args.in_context_num_samples,
        padding_side=args.padding_side,
        num_train_samples=args.num_train_samples,
        num_val_samples=args.num_val_samples,
        num_test_samples=args.num_test_samples,
        dpo_training=True if args.objective in ['dpo', 'cpo'] else False,
        add_synthetic_pref_data=args.synthetic_pref_data,
        only_synthetic=args.only_synthetic,
        remove_neutral_train_data=args.remove_neutral_train_data
    )

    if args.devices == 'auto':
        devices = 'auto'
    elif args.devices.startswith('['):
        devices = ast.literal_eval(args.devices)
    else:
        devices = int(args.devices)

    # Create trainer
    trainer, checkpoint_callback = create_trainer(
        experiment_name=args.experiment_name,
        accelerator=args.accelerator,
        devices=devices,
        # devices=[0],
        strategy=args.strategy,
        max_epochs=args.max_epochs,
        checkpoint_dir=checkpoint_dir,
        val_check_interval=args.val_check_interval,
        early_stop_patience=args.patience,
        log_model_mlflow=args.log_model_mlflow,
        lora_finetuning=args.lora_finetuning
    )

    if args.just_test:
        trainer.test(model, datamodule=datamodule)

    else:
        # Train the model
        trainer.fit(model, datamodule=datamodule)

        # if trainer.is_global_zero:
        # Reload best checkpoint and run test
        best_model_path = checkpoint_callback.best_model_path
        print(f"Best model path: {best_model_path}")

        model = custom_load_from_checkpoint(model, best_model_path, args, best_checkpoint=True)
        trainer.test(model, datamodule=datamodule)

    # Delete all files inside the checkpoint directory
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        print(f"Deleted all files in {checkpoint_dir}")


parser = argparse.ArgumentParser(description='Train a translation model')
parser.add_argument('--train-data', type=str, required=True,
                    help='Path to the training data')
parser.add_argument('--val-data', type=str, required=True,
                    help='Path to the validation data')
parser.add_argument('--test-data', type=str, required=True,
                    help='Path to the test data')
parser.add_argument('--experiment-name', type=str, required=True,
                    help='MLFlow experiment name')
parser.add_argument('--src-lng', type=str, required=True,
                    help='Source language')
parser.add_argument('--tgt-lng', type=str, required=True,
                    help='Target language')
parser.add_argument('--just-test', action='store_true',
                    help='Only test the model without training')
parser.add_argument('--model-name', type=str, required=True,
                    help='Model name for initialization from Hugging Face Checkpoint / or local path')
parser.add_argument('--from-pretrained', type=str, default=None,
                    help='Path to pretrained model checkpoint')
parser.add_argument('--from-lora', type=str, default=None,
                    help='Path to pretrained LoRA parameters checkpoint')
parser.add_argument("--overwrite-arguments", action="store_true",
                    help='Whether to overwrite arguments when loading a pre-trained model.')
parser.add_argument('--pivot-lng', type=str, default=None,
                    help='Load the translator model as a pivot model that translates into the pivot language')
parser.add_argument('--learning-rate', type=float, default=5e-5,
                    help='Learning rate for the optimizer')
parser.add_argument('--warmup-steps', type=int, default=0,
                    help='Number of warmup steps for the learning rate scheduler')
parser.add_argument('--weight-decay', type=float, default=0.00,
                    help='Weight decay for the optimizer')
# Arguments specific for the pivot model
parser.add_argument('--model-name-2', type=str, default=None,
                    help='Model name for initialization from Hugging Face Checkpoint / or local path'
                         'when using the translate pivot model')
parser.add_argument('--train-model-1', action='store_true',
                    help='Train model 1')
parser.add_argument('--train-model-2', action='store_true',
                    help='Train model 2')
# AWS Bedrock specific arguments
parser.add_argument('--s3-bucket', type=str, default=None,
                    help='S3 bucket for AWS Bedrock model')
parser.add_argument('--iam-role', type=str, default=None,
                    help='S3 prefix for AWS Bedrock model')
# Dataset specific arguments
parser.add_argument('--batch-size', type=int, default=4,
                    help='Batch size for training and inference')
parser.add_argument('--num-train-samples', type=int, default=None,
                    help='Number of training samples to use. Default is None (use all samples)')
parser.add_argument('--num-val-samples', type=int, default=None,
                    help='Number of validation samples to use. Default is None (use all samples)')
parser.add_argument('--num-test-samples', type=int, default=None,
                    help='Number of test samples to use. Default is None (use all samples)')
parser.add_argument('--max-length', type=int, default=1024,
                    help='Maximum length of the input sequence')
parser.add_argument('--padding-side', type=str, default='right',
                    help='Padding side for the input sequence. Options: "left" or "right"'
                         'Use "right" for encoder-decoder models and "left" for decoder-only models')
parser.add_argument('--prompt', type=str, default="<INPUT_SRC>",
                    help='Prompt for the model')
parser.add_argument('--in-context-learning', action='store_true',
                    help='Use in-context learning for the model')
parser.add_argument('--in-context-num-samples', type=int, default=1,
                    help='Number of in-context learning samples to use. Default is 1')
parser.add_argument('--synthetic-pref-data', type=str, nargs='+', default=None,
                    help='List of paths to additional synthetic training data.')
parser.add_argument('--only-synthetic', action='store_true',
                    help='Use only synthetice data to train the model')
parser.add_argument('--remove-neutral-train-data', action='store_true',
                    help='Remove neutral samples from training data.')
# Trainer specific arguments
parser.add_argument('--accelerator', type=str, default='auto',
                    help='Accelerator for training. Options: "cpu", "gpu", "tpu", "ipu", "hpu", "auto"')
parser.add_argument('--devices', type=str, default='auto',
                    help='Number of devices for training. Options: "auto", "0", "1", etc.')
parser.add_argument('--strategy', type=str, default='auto',
                    help='The training strategy')
parser.add_argument('--precision', type=str, default='16-mixed',
                    help='Precision for training. Options: "16", "32", "64", "bf16", "fp16", "mixed", "auto"')
parser.add_argument('--max-epochs', type=int, default=1,
                     help='Maximum number of epochs for training. Default is 1')
parser.add_argument('--val-check-interval', type=float, default=1.0,
                    help='Validation check interval. Default is 1.0 (validate every epoch).'
                         'If lower than 1 do it every fraction of steps per epoch, '
                         'if greater than 1 N number of steps.')
parser.add_argument('--patience', type=int, default=10,
                     help='Number of checkpoints to continue without validation improvement')
parser.add_argument('--log-model-mlflow', action='store_true',
                    help='Log the model to MLFlow after training')
parser.add_argument('--log-lora-mlflow', action='store_true',
                    help='Log only the LoRA parameters to MLFlow after training')
parser.add_argument('--fix-seed', type=int, default=None,
                    help='Fix the seed to be able to reproduce results')
# Generation specific arguments
parser.add_argument('--do-sample', type=bool, default=False,
                    help='Whether to sample from the model during generation. Default is False (use greedy decoding)')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='Temperature for sampling. Default is 1.0 (no temperature scaling)')
parser.add_argument('--top-k', type=int, default=50,
                    help='Top-k sampling for generation. Default is 50')
parser.add_argument('--top-p', type=float, default=0.90,
                    help='Top-p (nucleus) sampling for generation. Default is 0.95')
# Different objective function options
parser.add_argument('--objective', type=str, default='cross_entropy',
                    choices=['cross_entropy', 'dpo', 'cpo'],
                    help='Objective function for training. Default is "cross_entropy".')
parser.add_argument('--dpo-beta', type=float, default=0.1,
                    help='Beta value for DPO loss. Default is 0.1')
parser.add_argument('--dpo-label-smoothing', type=float, default=0.0,
                    help='Label smoothing value for DPO loss. Default is 0.0')
parser.add_argument('--cpo-lambda', type=float, default=1.0,
                    help='Lambda value for CPO loss. Lambda 1 means no NLL regularization. '
                         'while 0.5 means equal regularization. Default is 1.0')
parser.add_argument("--lora-finetuning", action="store_true",
                    help='Apply LoRA fine-tuning')
parser.add_argument("--bedrock-type", type=str, default='batch', choices=['batch', 'demand'],
                    help='Type of AWS Bedrock deployment. Default is "batch".')
if __name__ == '__main__':
    my_args = parser.parse_args()
    main(my_args)
