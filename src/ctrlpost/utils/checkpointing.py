import os
import json

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from peft import PeftModel
from pytorch_lightning.loggers import MLFlowLogger


class MLflowCustomCheckpointCallback(ModelCheckpoint):
    """Custom callback that saves LoRA adapters alongside MLflow checkpoints"""

    def __init__(self, dirpath, filename, save_top_k, monitor, mode, lora_finetuning, log_model_mlflow):
        super().__init__(dirpath=dirpath, filename=filename, save_top_k=save_top_k, monitor=monitor, mode=mode)

        self.lora_finetuning = lora_finetuning
        self.log_model_mlflow = log_model_mlflow

    def on_train_end(self, trainer, pl_module):


        num_steps = int(self.best_model_path.split('/')[-1].split('-')[2].strip('step='))
        # Multiply with batch size to get num samples
        if hasattr(trainer.model, 'module'):
            num_train_samples = trainer.model.module.hparams.batch_size * num_steps
        else:
            num_train_samples = trainer.model.hparams.batch_size * num_steps
        # log into MLFlow
        mlflow_logger = None
        for logger in trainer.loggers:
            if isinstance(logger, MLFlowLogger):
                mlflow_logger = logger
                break
        mlflow_logger.experiment.log_metric(
            mlflow_logger.run_id,
            'best_ckpt_num_train_steps',
            num_steps
        )
        mlflow_logger.experiment.log_metric(
            mlflow_logger.run_id,
            'best_ckpt_num_train_samples',
            num_train_samples
        )

        if self.lora_finetuning and self.log_model_mlflow:
            if self.best_model_path and os.path.exists(self.best_model_path):
                if mlflow_logger:
                    list_files = [file for file in os.listdir(self.best_model_path)]
                    for file in list_files:
                        file_obj_path = f'{self.best_model_path}/{file}'
                        mlflow_logger.experiment.log_artifact(
                            mlflow_logger.run_id,
                            file_obj_path,
                            self.best_model_path.split('/')[-1]
                        )
                    print(f"LoRA adapter logged to MLflow: {self.best_model_path}")
                else:
                    print('Warning: Failed to log LoRA adapter to MLFlow')
        else:
            # Call the original method
            super().on_train_end(trainer, pl_module)

    def _save_checkpoint(self, trainer, filepath):

        # Save LoRA adapter if the model uses LoRA
        if self.lora_finetuning:
            _ = self._save_lora_adapter(trainer.model, filepath)
        else:
            # Save the standard checkpoint first
            super()._save_checkpoint(trainer, filepath)

    def _save_lora_adapter(self, model, checkpoint_filepath):
        """Save LoRA adapter weights alongside the checkpoint"""

        # Extract LoRA adapter weights
        if hasattr(model, 'module'):
            actual_model = model.module
        else:
            actual_model = model

        if hasattr(actual_model.model, 'peft_config'):
            # Save adapter weights as separate file
            adapter_filedir = checkpoint_filepath.replace('.ckpt', '_LoRA')
            os.makedirs(adapter_filedir, exist_ok=True)

            actual_model.model.save_pretrained(adapter_filedir)

            self.best_model_path = adapter_filedir
            print(f"LoRA adapter saved to {adapter_filedir}")
            return adapter_filedir
        else:
            print("Warning: No LoRA adapter parameters found to save!")
            return None


def custom_load_from_checkpoint(model, checkpoint_path, args, strict=False, best_checkpoint=False):

    if checkpoint_path.endswith('.ckpt'):
        if args.overwrite_arguments:
            model = type(model).load_from_checkpoint(checkpoint_path, strict=strict, **vars(args))
        else:
            model = type(model).load_from_checkpoint(checkpoint_path, strict=strict, )
    elif checkpoint_path.endswith('_LoRA'):
        if best_checkpoint:
            # This means there is already an adapter in the model, need to replace it for the best one
            model.model.load_adapter(checkpoint_path, adapter_name='best_adapter')
            model.model.set_adapter('best_adapter')
        else:
            model.model = PeftModel.from_pretrained(model.model, checkpoint_path, is_trainable=True)
    else:
        raise(f'This checkpoint type is not supported: {checkpoint_path}')



    return model