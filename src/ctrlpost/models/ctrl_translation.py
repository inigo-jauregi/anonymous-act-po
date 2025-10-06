'''
Ctrl Translation baseline class for all translation and APE models
'''
import copy

import torch
import torch.nn.functional as F
from torchtune.rlhf.loss import DPOLoss
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModel
)
from typing import Dict, Any, Optional

from src.ctrlpost.models.base_model import BaseTranslationModel
from src.ctrlpost.utils.language_mappings import lang_token_mapping


class CtrlTranslationModel(BaseTranslationModel):
    def __init__(
            self,
            model_name: str,
            **kwargs
    ):
        """
        Base PyTorch Lightning model for translation and APE tasks.

        Args:
            model_name (str): Hugging Face model name/path
            **kwargs: Additional configuration parameters
        """

        tgt_lang_tkn = lang_token_mapping(kwargs['tgt_lng'], 'NLLB')

        # Load pretrained model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                  tgt_lang=tgt_lang_tkn)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        kwargs['generation_config'] = {
            'max_new_tokens': 512,
            'temperature': kwargs['temperature'],
            'do_sample': kwargs['do_sample'],
            'top_p': kwargs['top_p'],
            'eos_token_id': tokenizer.eos_token_id,
            # 'early_stopping': True,
            # 'no_repeat_ngram_size': 3,
            # 'repetition_penalty': 1.2,
            # 'length_penalty': 0.8
        }
        super().__init__( **kwargs)

        if kwargs['lora_finetuning']:
            model = self.add_lora_finetuning(model)
            self.lora_finetuning = True
        else:
            self.lora_finetuning = False

        self.tokenizer = tokenizer
        self.model = model

        self.model_type = 'encoder-decoder'
        self.lora_finetuning = False

    def forward(self, **batch):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token ids
            attention_mask (torch.Tensor): Attention mask for input
            target_ids (torch.Tensor, optional): Target token ids for training

        Returns:
            Model output
        """
        return self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['target_ids']
        )

    def compute_loss(self, outputs):
        """
        Simple transformers package loss (already computed internally).
        :param outputs:
        :return:
        """
        return outputs.loss

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: Optional[int] = None):
        """
        Prediction step for inference.

        Args:
            batch (Dict[str, torch.Tensor]): Batch of data
            batch_idx (Optional[int], optional): Batch index

        Returns:
            Generated predictions
        """

        bos_token_nllb = lang_token_mapping(self.tgt_lng, 'NLLB')

        preds = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(bos_token_nllb),
            # TODO: Potentially add return_dict_in_generate=True and output_scores=True (for CRPO calculations)
            **self.extra_config.get('generation_config', {})
        )

        return preds


class CtrlTranslationModelDpo(CtrlTranslationModel):
    def __init__(
            self,
            model_name: str,
            **kwargs
    ):
        """
        Base PyTorch Lightning model for translation and APE tasks.

        Args:
            model_name (str): Hugging Face model name/path
            **kwargs: Additional configuration parameters
        """
        super().__init__(model_name, **kwargs)

        # Create a reference to the model for DTO purposes
        self.model_ref = AutoModelForSeq2SeqLM.from_pretrained(model_name).eval()
        for param in self.model_ref.parameters():
            param.requires_grad = False

        # Initialize the DPOLoss
        self.dpo_loss = DPOLoss(beta=kwargs['dpo_beta'], label_smoothing=kwargs['dpo_label_smoothing'])

        # if a pretrained model being loaded we need to overwrite the model_ref with the pretrained model
        self.flag_overwrite_ref_model_parameters = True


    def forward(self, **batch):
        """
        Forward pass of the model for DTO.

        Args:
            input_ids (torch.Tensor): Input token ids
            attention_mask (torch.Tensor): Attention mask for input
            target_ids (torch.Tensor): Target token ids for training
            target_ids_contrastive (torch.Tensor): Target token ids for contrastive output
            target_lengths (torch.Tensor): Lengths of target sequences
            formality_labels (str): Labels for formality classification
            target_annotated_text (str): Annotated target text
            padding_side (str): Padding side for the model
            target_annotated_text_contrastive (str): Annotated target text for contrastive output
            target_ids_contrastive (torch.Tensor): Target token ids for contrastive output
            target_attention_mask_contrastive (torch.Tensor): Attention mask for contrastive output
            target_lengths_contrastive (torch.Tensor): Lengths of target sequences for contrastive output

        Returns:
            Model output
        """
        preferred_logits =  self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['target_ids']
        ).logits
        preferred_logprobs = self.get_log_prob(
            preferred_logits,
            batch['target_ids'],
            batch['target_attention_mask']
        )

        rejected_logits = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['target_ids_contrastive']
        ).logits
        rejected_logprobs = self.get_log_prob(
            rejected_logits,
            batch['target_ids_contrastive'],
            batch['target_attention_mask_contrastive']
        )

        with torch.no_grad():
            ref_preferred_logits = self.model_ref(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['target_ids']
            ).logits
            ref_preferred_logprobs = self.get_log_prob(
                ref_preferred_logits,
                batch['target_ids'],
                batch['target_attention_mask']
            )
            ref_rejected_logits = self.model_ref(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['target_ids_contrastive']
            ).logits
            ref_rejected_logprobs = self.get_log_prob(
                ref_rejected_logits,
                batch['target_ids_contrastive'],
                batch['target_attention_mask_contrastive']
            )


        return {
            'preferred_logprobs': preferred_logprobs,
            'rejected_logprobs': rejected_logprobs,
            'ref_preferred_logprobs': ref_preferred_logprobs,
            'ref_rejected_logprobs': ref_rejected_logprobs
        }

    def compute_loss(self, outputs):
        """
        Compute the loss based on the preferred and rejected log probabilities.

        Args:
            outputs (Dict[str, torch.Tensor]): Model outputs containing log probabilities

        Returns:
            torch.Tensor: Computed loss value
        """

        loss_output = self.dpo_loss(
            policy_chosen_logps=outputs['preferred_logprobs'],
            policy_rejected_logps=outputs['rejected_logprobs'],
            reference_chosen_logps=outputs['ref_preferred_logprobs'],
            reference_rejected_logps=outputs['ref_rejected_logprobs']
        )

        # Average of the loss across the batch
        loss = loss_output[0].mean()
        # chosen_rewards
        avg_chosen_rewards = loss_output[1].mean()
        self.log(f'{self.eval_log_type}_avg_chosen_rewards', avg_chosen_rewards,
                 on_step=True, on_epoch=True, prog_bar=True)
        # rejected_rewards
        avg_rejected_rewards = loss_output[2].mean()
        self.log(f'{self.eval_log_type}_avg_rejected_rewards', avg_rejected_rewards,
                 on_step=True, on_epoch=True, prog_bar=True)
        # reward margin
        avg_reward_margin = avg_chosen_rewards - avg_rejected_rewards
        self.log(f'{self.eval_log_type}_avg_reward_margin', avg_reward_margin,
                 on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def overwrite_ref_model_parameters(self):

        # Debugging
        # print("Original model:")
        # for name, param in self.model.named_parameters():
        #     print(f"{name}: {param.shape}")
        # print("Ref model:")
        # for name, param in self.model_ref.named_parameters():
        #     print(f"{name}: {param.shape}")

        print("Initializing reference model with pretrained weights...")

        # Ger the config and create a new model instance
        # config = self.model.config
        # self.model_ref = AutoModel.from_config(config)

        # self.model_ref.load_state_dict(self.model.state_dict()).eval()
        self.model_ref = copy.deepcopy(self.model).eval()
        for param in self.model_ref.parameters():
            param.requires_grad = False


class CtrlTranslationModelCpo(CtrlTranslationModel):
    def __init__(
            self,
            model_name: str,
            **kwargs
    ):
        """
        Pytorch class to train NLLB model with the CPO algorithm.

        Args:
            model_name (str): Hugging Face model name/path
            **kwargs: Additional configuration parameters
        """
        super().__init__(model_name, **kwargs)

        self.cpo_beta = torch.tensor(kwargs.get('dpo_beta', 0.1))
        self.cpo_label_smoothing = kwargs.get('dpo_label_smoothing', 0.0)
        self.cpo_lambda = kwargs.get('cpo_lambda', 1.0)

        # if a pretrained model being loaded we need to overwrite the model_ref with the pretrained model
        self.flag_overwrite_ref_model_parameters = True


    def forward(self, **batch):
        """
        Forward pass of the model for DTO.

        Args:
            input_ids (torch.Tensor): Input token ids
            attention_mask (torch.Tensor): Attention mask for input
            target_ids (torch.Tensor): Target token ids for training
            target_ids_contrastive (torch.Tensor): Target token ids for contrastive output
            target_lengths (torch.Tensor): Lengths of target sequences
            formality_labels (str): Labels for formality classification
            target_annotated_text (str): Annotated target text
            padding_side (str): Padding side for the model
            target_annotated_text_contrastive (str): Annotated target text for contrastive output
            target_ids_contrastive (torch.Tensor): Target token ids for contrastive output
            target_attention_mask_contrastive (torch.Tensor): Attention mask for contrastive output
            target_lengths_contrastive (torch.Tensor): Lengths of target sequences for contrastive output

        Returns:
            Model output
        """
        preferred_output =  self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['target_ids']
        )
        preferred_logits = preferred_output.logits
        preferred_loss = preferred_output.loss
        preferred_logprobs = self.get_log_prob(
            preferred_logits,
            batch['target_ids'],
            batch['target_attention_mask']
        )

        rejected_logits = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['target_ids_contrastive']
        ).logits
        rejected_logprobs = self.get_log_prob(
            rejected_logits,
            batch['target_ids_contrastive'],
            batch['target_attention_mask_contrastive']
        )


        return {
            'preferred_logprobs': preferred_logprobs,
            'rejected_logprobs': rejected_logprobs,
            'preferred_loss': preferred_loss
        }

    def compute_loss(self, outputs):
        """
        Compute the CPO loss based on the preferred and rejected log probabilities.

        Args:
            outputs (Dict[str, torch.Tensor]): Model outputs containing log probabilities

        Returns:
            torch.Tensor: Computed loss value
        """


        # Copying the 'sigmoid' implemetation from
        # https://github.com/huggingface/trl/blob/main/trl/trainer/cpo_trainer.py
        logits = (outputs['preferred_logprobs'] - outputs['rejected_logprobs'])
        # This reduces to Equation 3 from the CPO paper when label_smoothing is 0
        losses_prefer = (-F.logsigmoid(self.cpo_beta * logits) * (1 - self.cpo_label_smoothing)
                         -F.logsigmoid(-self.cpo_beta * logits) * self.cpo_label_smoothing)
        loss_prefer = torch.mean(losses_prefer)
        # In the CPO paper they add a NLL loss over the preferred output
        # NOTE: for some reason in TRL they don't add the NLL loss (WHYYY???)
        cpo_loss = (self.cpo_lambda * loss_prefer +
                   (torch.tensor(1.0, device=self.device) - self.cpo_lambda) * outputs['preferred_loss'])

        # chosen_rewards
        chosen_rewards = self.cpo_beta * outputs['preferred_logprobs']
        avg_chosen_rewards = chosen_rewards.mean().detach()
        self.log(f'{self.eval_log_type}_avg_chosen_rewards', avg_chosen_rewards,
                 on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        # rejected_rewards
        avg_rejected_rewards = self.cpo_beta * outputs['rejected_logprobs']
        avg_rejected_rewards = avg_rejected_rewards.mean().detach()
        self.log(f'{self.eval_log_type}_avg_rejected_rewards', avg_rejected_rewards,
                 on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        # reward margin
        avg_reward_margin = avg_chosen_rewards - avg_rejected_rewards
        self.log(f'{self.eval_log_type}_avg_reward_margin', avg_reward_margin,
                 on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return cpo_loss
