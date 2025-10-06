'''
Base class for all translation and APE models
'''
import re
import tempfile
import os
import shutil

import pandas as pd
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from transformers import (
    get_linear_schedule_with_warmup
)
from typing import Dict, Optional
from peft import PeftModel, LoraConfig, get_peft_model, TaskType

import sacrebleu
from sacrebleu.metrics import BLEU
from comet import download_model, load_from_checkpoint

from src.ctrlpost.utils import evaluation


class BaseTranslationModel(pl.LightningModule):
    def __init__(self, **kwargs):
        """
        Base PyTorch Lightning model for translation and APE tasks.

        Args:
            src_lang (str): Source language
            tgt_lang (str): Target language
            learning_rate (float, optional): Learning rate for optimizer. Defaults to 5e-5.
            warmup_steps (int, optional): Number of warmup steps for learning rate scheduler. Defaults to 0.
            weight_decay (float, optional): Weight decay for optimizer. Defaults to 0.0.
            **kwargs: Additional configuration parameters
        """
        super().__init__()

        # Save hyperparameters for logging and checkpoint
        self.save_hyperparameters()

        # Additional configurable parameters
        self.learning_rate = kwargs['learning_rate']
        self.warmup_steps = kwargs['warmup_steps']
        self.weight_decay = kwargs['weight_decay']

        # Store any additional kwargs for potential custom configurations
        self.extra_config = kwargs

        self.tokenizer = None  # Not implemented in the base class, but all models should have one
        self.model = None  # Not implemented in the base class, but all models should have one

        self.src_lng = kwargs['src_lng']
        self.tgt_lng = kwargs['tgt_lng']

        # Save validation predictions
        self.val_predictions = []
        self.eval_log_type = "val"  # Default evaluation log type

        # Metrics
        self.comet_model_name = 'Unbabel/wmt22-comet-da'
        self.comet_model = load_from_checkpoint(download_model(self.comet_model_name))
        self.comet_model.eval()
        for param in self.comet_model.parameters():
            param.requires_grad = False

    def add_lora_finetuning(self, model):

        self.lora_finetuning = True
        # Default LoRA configuration
        self.default_lora_config = {
            "peft_type": "LORA",
            "task_type": TaskType.CAUSAL_LM,
            "inference_mode": False,
            "r": 16,  # rank
            "lora_alpha": 32,  # scaling parameter
            "lora_dropout": 0.1,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],  # Common for Qwen models
            "bias": "none",
            "modules_to_save": None,
        }

        # # Update with user-provided config
        # if lora_config:
        #     default_lora_config.update(lora_config)

        # Create LoRA config and apply to model
        peft_config = LoraConfig(**self.default_lora_config)
        model = get_peft_model(model, peft_config)

        # Print trainable parameters info
        print("LoRA finetuning enabled. Trainable parameters:")
        model.print_trainable_parameters()

        return model

    def forward(self, **batch):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token ids
            attention_mask (torch.Tensor): Attention mask for input
            labels (torch.Tensor, optional): Target token ids for training

        Returns:
            Model output
        """
        return NotImplementedError

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step for the model.

        Args:
            batch (Dict[str, torch.Tensor]): Batch of data
            batch_idx (int): Batch index

        Returns:
            torch.Tensor: Computed loss
        """

        self.eval_log_type = 'train'

        outputs = self(**batch)
        loss = self.compute_loss(outputs)

        # Log training loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """
        Validation step for the model.

        Args:
            batch (Dict[str, torch.Tensor]): Batch of data
            batch_idx (int): Batch index
        """

        self.eval_log_type = 'val' if self.eval_log_type == 'train' else self.eval_log_type

        outputs = self(**batch)
        val_loss = self.compute_loss(outputs)

        # Log validation loss
        self.log(f'{self.eval_log_type}_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True,
                 sync_dist=True)

        # Generate predictions
        preds = self.predict_step(batch)

        self.val_predictions.append(
            {
                'preds': preds,
                'gt': batch['target_ids'],
                'contrastive': batch['target_text_contrastive'],
                'raw_source_text': batch['raw_source_text'],
                'input_ids': batch['input_ids'],
                'gt_annotated': batch['target_annotated_text'],
                'gt_annotated_contrastive': batch['target_annotated_text_contrastive']
                                            if 'target_annotated_text_contrastive' in batch else None,
                'attribute_label': batch['attribute_labels']
            }
        )

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """
        Test step for the model.

        Args:
            batch (Dict[str, torch.Tensor]): Batch of data
            batch_idx (int): Batch index
        """
        self.eval_log_type = 'test'
        self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.

        Returns:
            Tuple of optimizer and learning rate scheduler
        """
        if self.lora_finetuning:
            # Only optimize trainable parameters (LoRA adapters)
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        else:
            # Prepare optimizer
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )

        # Prepare learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: Optional[int] = None):
        """
        Prediction step for inference.

        Args:
            batch (Dict[str, torch.Tensor]): Batch of data
            batch_idx (Optional[int], optional): Batch index

        Returns:
            Generated predictions
        """
        # return self.model.generate(
        #     input_ids=batch['input_ids'],
        #     attention_mask=batch['attention_mask'],
        #     **self.extra_config.get('generation_config', {})
        # )
        return NotImplementedError

    def on_validation_epoch_end(self):
        """
        Optional method to log learning rate at the end of each validation epoch
        """

        lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', lr)

        # Gather predictions from all GPUs
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

        if world_size > 1:
            all_predictions = [None for _ in range(world_size)]

            # Gather data from all processes
            torch.distributed.all_gather_object(all_predictions, self.val_predictions)

            if self.global_rank == 0:
                all_predictions = [pred for preds in all_predictions for pred in preds]
                prep_all_predictions = self.prepare_predictions(all_predictions)
                # Calculate scores
                dict_scores = evaluation(prep_all_predictions, self.tgt_lng,
                                         comet_model=self.comet_model, loggers=self.loggers)
                # dict_scores = self.evaluation_scores(all_predictions, world_size=world_size)
            else:
                all_predictions = [pred for preds in all_predictions for pred in preds]
                prep_all_predictions = self.prepare_predictions(all_predictions)
                dict_scores = evaluation(prep_all_predictions, self.tgt_lng, empty_return=True,
                                         comet_model=self.comet_model, loggers=self.loggers)
                # dict_scores = self.evaluation_scores(all_predictions, empty_return=True)

            for key, value in dict_scores.items():
                # Broadcast the BLEU score to all ranks so they all have the same value
                value_tensor = torch.tensor(value, device=self.device, dtype=torch.float32)
                torch.distributed.broadcast(value_tensor, src=0)
                self.log(f'{self.eval_log_type}_{key}', value_tensor, sync_dist=True, rank_zero_only=True)
                if self.global_rank == 0:
                    print(f'{self.eval_log_type}_{key}: ', value)
        else:
            all_predictions = self.val_predictions
            prep_all_predictions = self.prepare_predictions(all_predictions)
            dict_scores = evaluation(prep_all_predictions, self.tgt_lng,
                                     comet_model=self.comet_model, loggers=self.loggers)
            # dict_scores = self.evaluation_scores(all_predictions)
            for key, value in dict_scores.items():
                self.log(f'{self.eval_log_type}_{key}', value)
                print(f'{self.eval_log_type}_{key}: ', value)

        self.val_predictions = []

    def on_test_epoch_end(self):
        """
        Optional method to log learning rate at the end of each test epoch
        """

        # Gather predictions from all GPUs
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

        if world_size > 1:
            all_predictions = [None for _ in range(world_size)]

            # Gather data from all processes
            torch.distributed.all_gather_object(all_predictions, self.val_predictions)

            if self.global_rank == 0:
                all_predictions = [pred for preds in all_predictions for pred in preds]
                prep_all_predictions = self.prepare_predictions(all_predictions)
                # Calculate scores
                dict_scores = evaluation(prep_all_predictions, self.tgt_lng, save_preds=True,
                                         comet_model=self.comet_model, loggers=self.loggers)
                # dict_scores = self.evaluation_scores(all_predictions, world_size=world_size)
            else:
                all_predictions = [pred for preds in all_predictions for pred in preds]
                prep_all_predictions = self.prepare_predictions(all_predictions)
                dict_scores = evaluation(prep_all_predictions, self.tgt_lng, empty_return=True)
                # dict_scores = self.evaluation_scores(all_predictions, empty_return=True)

            for key, value in dict_scores.items():
                # Broadcast the BLEU score to all ranks so they all have the same value
                value_tensor = torch.tensor(value, device=self.device, dtype=torch.float32)
                torch.distributed.broadcast(value_tensor, src=0)
                self.log(f'{self.eval_log_type}_{key}', value_tensor)

        else:
            all_predictions = self.val_predictions
            prep_all_predictions = self.prepare_predictions(all_predictions)
            dict_scores = evaluation(prep_all_predictions, self.tgt_lng, save_preds=True,
                                     comet_model=self.comet_model, loggers=self.loggers)
            # dict_scores = self.evaluation_scores(all_predictions, save_preds=True)
            for key, value in dict_scores.items():
                self.log(f'{self.eval_log_type}_{key}', value)
                print(f'{self.eval_log_type}_{key}: ', value)

        self.val_predictions = []
        self.eval_log_type = 'val'  # Reset evaluation log type to default after test is done

    def evaluation_scores(self, all_predictions, save_preds=False, world_size=1, empty_return=False):

        dict_scores = {}
        prep_all_predictions = self.prepare_predictions(all_predictions)

        if empty_return:
            # Get unique attribute labels
            unique_labels = set(prep_all_predictions['attribute_label_list'])
            dict_scores['bleu'] = 0.0
            dict_scores['comet_score'] = 0.0
            for label in unique_labels:
                dict_scores[f'{label}_m_acc'] = 0.0
                dict_scores[f'{label}_strict_m_acc'] = 0.0
                dict_scores[f'{label}_token_recall'] = 0.0
            dict_scores['avg_m_acc'] = 0.0
            dict_scores['avg_strict_m_acc'] = 0.0
            dict_scores['avg_token_recall'] = 0.0
            return dict_scores

        # Compute BLEU score
        bleu = sacrebleu.corpus_bleu(prep_all_predictions['prediction_list'],
                                     [prep_all_predictions['gt_list']],
                                     tokenize="ja-mecab" if self.tgt_lng in ['ja', 'zh'] else BLEU.TOKENIZER_DEFAULT)
        dict_scores['bleu'] = bleu.score
        # bleu_tensor = torch.tensor(bleu.score, device=self.device)
        # if world_size > 1:
        #     # Ensure all ranks participate in barrier
        #     if torch.distributed.is_initialized():
        #         torch.distributed.barrier()
        #         print('This happens!')
        #         torch.distributed.broadcast(bleu_tensor, src=0)
        # self.log(f'{self.eval_log_type}_bleu', bleu_tensor)
        # print('bleu:', bleu.score)

        # Compute COMET score
        comet_score = self.comet_model.predict(prep_all_predictions['data_comet'],
                                               batch_size=8).system_score
        # comet_score = 100 * comet_score
        # self.log(f'{self.eval_log_type}_comet_score', comet_score, rank_zero_only=True)
        # print('COMET: ', comet_score)
        dict_scores['comet_score'] = comet_score

        # Matched Accuracy
        if len(prep_all_predictions['gt_annotated_list_contrastive']) > 0:
            dict_label_scores_m_acc = {}
            dict_label_scores_strict_m_acc = {}
            dict_label_scores_token_recall = {}
            for label, pred_text, gt_annotated, gt_annotated_contr in (
                    zip(prep_all_predictions['attribute_label_list'],
                        prep_all_predictions['prediction_list'],
                        prep_all_predictions['gt_annotated_list'],
                        prep_all_predictions['gt_annotated_list_contrastive'])):
                # Identify annotated markers in the gt_annotated text
                matches = re.findall(r'\[F\](.*?)\[\/F\]', gt_annotated)
                not_to_match = re.findall(r'\[F\](.*?)\[\/F\]', gt_annotated_contr)
                # if self.tgt_lng not in ['ja', 'zh']:
                #     # Add spaces around the matches to avoid partial matches
                #     # except for written languages with not many spaces
                #     matches = [f" {match} " for match in matches]
                #     not_to_match = [f" {match} " for match in not_to_match]

                # A match to the label if
                # 1. It matches with at least one of the 'matches'
                # 2. It does not match with any of the 'not_to_match
                matched_label = False
                for match in matches:
                    if self.tgt_lng in ['ja', 'zh']:
                        pattern = re.escape(match)
                    else:
                        pattern = r'\b' + re.escape(match) + r'\b'
                    if re.search(pattern, pred_text):
                        matched_label = True
                        break
                matched_contrastive_label = False
                for match in not_to_match:
                    if self.tgt_lng in ['ja', 'zh']:
                        pattern = re.escape(match)
                    else:
                        pattern = r'\b' + re.escape(match) + r'\b'
                    if re.search(pattern, pred_text):
                        matched_contrastive_label = True
                        break
                ########## COCOA MT M-Acc ##########
                if matched_label and not matched_contrastive_label:
                    if label not in dict_label_scores_m_acc:
                        dict_label_scores_m_acc[label] = {'correct': 0, 'total': 0}
                    dict_label_scores_m_acc[label]['correct'] += 1
                    dict_label_scores_m_acc[label]['total'] += 1
                if matched_contrastive_label and not matched_label:
                    if label not in dict_label_scores_m_acc:
                        dict_label_scores_m_acc[label] = {'correct': 0, 'total': 0}
                    dict_label_scores_m_acc[label]['total'] += 1
                ####################################

                ########## Strict M-Acc (or recall) ############
                # A variant of M-Acc where a neutral prediction is also considered incorrect
                # if the ground truth is not neutral
                if matched_label and not matched_contrastive_label:
                    if label not in dict_label_scores_strict_m_acc:
                        dict_label_scores_strict_m_acc[label] = {'correct': 0, 'total': 0}
                    dict_label_scores_strict_m_acc[label]['correct'] += 1
                    dict_label_scores_strict_m_acc[label]['total'] += 1
                if (not matched_label or matched_label and matched_contrastive_label) and len(matches) > 0:
                    if label not in dict_label_scores_strict_m_acc:
                        dict_label_scores_strict_m_acc[label] = {'correct': 0, 'total': 0}
                    dict_label_scores_strict_m_acc[label]['total'] += 1
                #####################################

                ########## Token-level Recall ##################
                # Metric that measures how many of the annotated tokens are present in the prediction, divided
                # by the total number of annotated tokens in the intended attribute class
                num_gt_tokens = len(matches)
                num_matched_tokens = 0
                for match in matches:
                    if match in pred_text:
                        num_matched_tokens += 1
                if label not in dict_label_scores_token_recall:
                    dict_label_scores_token_recall[label] = {'matched_tokens': 0, 'total': 0}
                dict_label_scores_token_recall[label]['matched_tokens'] += num_matched_tokens
                dict_label_scores_token_recall[label]['total'] += num_gt_tokens
                ################################################

            if len(dict_label_scores_m_acc) > 0:
                avg_acc = []
                for key, val in dict_label_scores_m_acc.items():
                    label_acc = 100 * val['correct'] / val['total']
                    # self.log(f'{self.eval_log_type}_{key}_m_acc', label_acc, rank_zero_only=True)
                    dict_scores[f'{key}_m_acc'] = label_acc
                    # print(f'{key}_m_acc: ', label_acc)
                    avg_acc.append(label_acc)
                avg_acc_val = sum(avg_acc) / len(avg_acc)
                dict_scores['avg_m_acc'] = avg_acc_val
                # self.log(f'{self.eval_log_type}_avg_m_acc', avg_acc_val, rank_zero_only=True)
            else:
                print('No M-Acc calculated')

            if len(dict_label_scores_strict_m_acc) > 0:
                avg_acc = []
                for key, val in dict_label_scores_strict_m_acc.items():
                    label_strict_m_acc = 100 * val['correct'] / val['total']
                    dict_scores[f'{key}_strict_m_acc'] = label_strict_m_acc
                    # self.log(f'{self.eval_log_type}_{key}_strict_m_acc', label_strict_m_acc, rank_zero_only=True)
                    # print(f'{key}_strict_m_acc: ', label_strict_m_acc)
                    avg_acc.append(label_strict_m_acc)
                avg_acc_val = sum(avg_acc) / len(avg_acc)
                dict_scores['avg_strict_m_acc'] = avg_acc_val
                # self.log(f'{self.eval_log_type}_avg_strict_m_acc', avg_acc_val, rank_zero_only=True)
            else:
                print('No Strict M-Acc calculated')

            if len(dict_label_scores_token_recall) > 0:
                avg_acc = []
                for key, val in dict_label_scores_token_recall.items():
                    label_token_recall = 100 * val['matched_tokens'] / val['total']
                    dict_scores[f'{key}_token_recall'] = label_token_recall
                    # self.log(f'{self.eval_log_type}_{key}_token_recall', label_token_recall, rank_zero_only=True)
                    # print(f'{key}_token_recall: ', label_token_recall)
                    avg_acc.append(label_token_recall)
                avg_acc_val = sum(avg_acc) / len(avg_acc)
                dict_scores['avg_token_recall'] = avg_acc_val
                # self.log(f'{self.eval_log_type}_avg_token_recall', avg_acc_val, rank_zero_only=True)
            else:
                print(f'No Token Recall calculated')

        if save_preds:
            index_list = list(range(len(prep_all_predictions['src_list'])))
            df_out = pd.DataFrame()
            df_out['index'] = index_list
            df_out['source'] = prep_all_predictions['src_raw_list']
            df_out['prompt_source'] = prep_all_predictions['src_list']  # Assuming source is the prompt
            df_out['attribute_label'] = prep_all_predictions['attribute_label_list']
            df_out['reference'] = prep_all_predictions['gt_list']
            df_out['reference_annotated'] = prep_all_predictions['gt_annotated_list']
            df_out['contrastive'] = prep_all_predictions['contrastive_list']
            df_out['contrastive_annotated'] = prep_all_predictions['gt_annotated_list_contrastive']
            df_out['prediction'] = prep_all_predictions['prediction_list']
            df_out.to_csv('predictions.csv', index=False)
            print('predictions saved to predictions.csv')

            # Log the artifact to MLflow
            mlflow_logger = None
            for logger in self.loggers:
                if isinstance(logger, MLFlowLogger):
                    mlflow_logger = logger
                    break

            if mlflow_logger is not None:
                mlflow_logger.experiment.log_artifact(
                    mlflow_logger.run_id,
                    'predictions.csv',
                    "predictions"
                )

        return dict_scores

    def prepare_predictions(self, all_predictions):
        prediction_list = []
        src_list = []
        src_raw_list = []
        gt_list = []
        gt_annotated_list = []
        contrastive_list = []
        gt_annotated_list_contrastive = []
        attribute_label_list = []
        data_comet = []
        for preds in all_predictions:
            # Decode each sentence (input, prediction, and ground truth)
            src_text = self.tokenizer.batch_decode(preds['input_ids'], skip_special_tokens=True)
            # print('\nSRC')
            # print(src_text[0])
            pred_text = self.tokenizer.batch_decode(preds['preds'], skip_special_tokens=True)
            if hasattr(self, 'rm_prompt_at_decoding'):
                if self.rm_prompt_at_decoding:
                    new_list_pred = []
                    for src_sample, pred_sample in zip(src_text, pred_text):
                        pred_sample = self.get_translation_only(src_sample, pred_sample)
                        # Further cleanup - stop at the first occurrence of any stop token
                        if '}' in pred_sample:
                            first_stop = pred_sample.index('}')
                            pred_sample = pred_sample[:first_stop]
                        new_list_pred.append(pred_sample)
                    pred_text = new_list_pred[:]
            # print('\nTGT')
            # print(pred_text[0])
            raw_source_text = preds['raw_source_text']
            src_raw_list.extend(raw_source_text)
            gt_text = self.tokenizer.batch_decode(preds['gt'], skip_special_tokens=True)
            gt_annotated_text = preds['gt_annotated']
            attribute_label = preds['attribute_label']
            prediction_list.extend(pred_text)
            src_list.extend(src_text)
            gt_list.extend(gt_text)
            gt_annotated_list.extend(gt_annotated_text)
            if preds['gt_annotated_contrastive']:
                contrastive_list.extend(preds['contrastive'])
                gt_annotated_text_contrastive = preds['gt_annotated_contrastive']
                gt_annotated_list_contrastive.extend(gt_annotated_text_contrastive)
            attribute_label_list.extend(attribute_label)
            for src, mt, ref in zip(raw_source_text, pred_text, gt_text):
                data_comet.append({'src': src, 'mt': mt, 'ref': ref})

        prep_dict = {
            'prediction_list': prediction_list,
            'src_raw_list': src_raw_list,
            'src_list': src_list,
            'gt_list': gt_list,
            'gt_annotated_list': gt_annotated_list,
            'contrastive_list': contrastive_list,
            'gt_annotated_list_contrastive': gt_annotated_list_contrastive,
            'attribute_label_list': attribute_label_list,
            'data_comet': data_comet
        }

        return prep_dict

    @staticmethod
    def get_translation_only(input_text, output_text):
        """Extracts only the generated translation, removing the input prompt."""
        # Remove the input prompt from the output
        if output_text.startswith(input_text):
            return output_text[len(input_text):].strip()
        return output_text.strip()

    @staticmethod
    def get_log_prob(logits, labels, mask):
        log_probs = F.log_softmax(logits, dim=-1)
        labels_clamp = torch.clamp(labels, min=0)  # in case the pad token is negative
        token_log_probs = torch.gather(log_probs, -1, labels_clamp.unsqueeze(-1)).squeeze(-1)

        response_log_probs = (token_log_probs * mask).sum(dim=-1)
        response_lengths = mask.sum(dim=-1).clamp(min=1)

        return response_log_probs / response_lengths
