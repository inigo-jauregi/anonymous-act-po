import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from typing import Union

from src.ctrlpost.utils import MLflowCustomCheckpointCallback


def create_trainer(
        experiment_name: str,
        max_epochs: int = 10,
        accelerator: str = 'auto',
        devices: Union[int, str] = 'auto',
        strategy: str = 'auto',
        precision: Union[str, int] = '16-mixed',
        log_dir: str = './lightning_logs',
        checkpoint_dir: str = './checkpoints',
        early_stop_patience: int = 3,
        val_check_interval: [int, float] = 1.0,
        log_model_mlflow: bool = False,
        lora_finetuning: bool = False
) -> pl.Trainer:
    """
    Create a configured PyTorch Lightning Trainer.

    Args:
        max_epochs (int, optional): Maximum number of training epochs. Defaults to 10.
        accelerator (str, optional): Training accelerator. Defaults to 'auto'.
        devices (Union[int, str], optional): Number or type of devices. Defaults to 'auto'.
        precision (Union[str, int], optional): Training precision. Defaults to '16-mixed'.
        log_dir (str, optional): Logging directory. Defaults to './lightning_logs'.
        checkpoint_dir (str, optional): Checkpoint directory. Defaults to './checkpoints'.
        wandb_project (Optional[str], optional): Weights & Biases project name. Defaults to None.
        early_stop_patience (int, optional): Patience for early stopping. Defaults to 3.
        val_check_interval (Union[int, float], optional): Validation check interval. Defaults to 1.0.
        log_model_mlflow (bool, optional): Whether to log the model in MLFlow. Defaults to False.

    Returns:
        pl.Trainer: Configured PyTorch Lightning Trainer
    """
    # Ensure directories exist
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Callbacks
    callbacks = [
        # Model checkpointing
        MLflowCustomCheckpointCallback(
            dirpath=checkpoint_dir,
            filename='model-{epoch:d}-{step}-{val_m_acc_bleu:.4f}',
            save_top_k=1,
            monitor='val_m_acc_bleu',
            mode='max',
            lora_finetuning=lora_finetuning,
            log_model_mlflow=log_model_mlflow
        ),
        # Early stopping
        pl.callbacks.EarlyStopping(
            monitor='val_m_acc_bleu',
            patience=early_stop_patience,
            mode='max'
        ),
        # Learning rate monitoring
        pl.callbacks.LearningRateMonitor(logging_interval='step'),
        # Rich progress bar
        # RichProgressBar(),
    ]

    # Loggers
    loggers = [
        # TensorBoard logger
        # pl.loggers.TensorBoardLogger(
        #     save_dir=log_dir,
        #     name='translation_model'
        # )
        # Mlflow logger
        MLFlowLogger(
            experiment_name=experiment_name,
            log_model=log_model_mlflow
            )
    ]

    # Define strategy
    if strategy == 'ddp':
        strategy = pl.strategies.DDPStrategy(
            broadcast_buffers=False,      # Disable buffer broadcasting entirely (it is giving me errors in multi-gpu setting)
        )

    # Convert val_check_interval to in if it is > 1.0
    if val_check_interval > 1.0:
        val_check_interval = int(val_check_interval)

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=20,
        val_check_interval=val_check_interval,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    return trainer, callbacks[0]  # return ModelCheckpoint callback
