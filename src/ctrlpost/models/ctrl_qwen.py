'''
CtrlQwen baseline class. Class for baseline Qwen multi-lingual LLMs
'''
import copy

import torch
import torch.nn.functional as F
from torchtune.rlhf.loss import DPOLoss
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from typing import Dict, Optional

from src.ctrlpost.models.base_model import BaseTranslationModel


class CtrlQwenModel(BaseTranslationModel):
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

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     torch_dtype=torch.float16,
                                                     # device_map="auto",
                                                     # offload_folder="./model_offload_cache",
                                                     # load_in_4bit=True,
                                                     # llm_int8_enable_fp32_cpu_offload=True,
                                                     # quantization_config={
                                                     #     "load_in_4bit": True,
                                                     #     "llm_int8_enable_fp32_cpu_offload": True,
                                                     #     "bnb_4bit_compute_dtype": torch.float16,
                                                     #     "bnb_4bit_use_double_quant": True,
                                                     #     "bnb_4bit_quant_type": "nf4"
                                                     # }
                                                     )

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
        super().__init__(**kwargs)

        if kwargs['lora_finetuning'] and kwargs['from_pretrained'] is None:
            model = self.add_lora_finetuning(model)
            self.lora_finetuning = True
        else:
            self.lora_finetuning = False

        # Load pretrained model and tokenizer
        self.tokenizer = tokenizer
        self.model = model

        self.rm_prompt_at_decoding = True
        self.model_type = 'decoder-only'

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
            input_ids=batch['input_ids_do'],
            attention_mask=batch['attention_mask_do'],
            labels=batch['target_ids_do']
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


        preds = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            # TODO: Potentially add return_dict_in_generate=True and output_scores=True (for CRPO calculations)
            **self.extra_config.get('generation_config', {})
        )

        return preds

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
        # preds = self.predict_step(batch)
        # print(outputs)
        logits = outputs['logits']

        # Get predictions by taking argmax
        preds = torch.argmax(logits, dim=-1)

        # print(batch)
        preds = self.mask_after_eos(
            preds,
            batch['target_ids_do'],
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id
        )

        # preds = preds[valid_mask]

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
        Validation step for the model.

        Args:
            batch (Dict[str, torch.Tensor]): Batch of data
            batch_idx (int): Batch index
        """

        self.eval_log_type = 'test'

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

    @staticmethod
    def mask_after_eos(preds, labels, eos_token_id, pad_token_id):
        """Mask predictions and labels after first EOS token"""
        batch_size, seq_len = preds.shape

        # Create mask for valid positions (before EOS)
        # valid_mask = torch.ones_like(preds, dtype=torch.bool)

        for i in range(batch_size):
            # Find first EOS in predictions
            # pred_eos_positions = (preds[i] == eos_token_id).nonzero(as_tuple=True)[0]
            # if len(pred_eos_positions) > 0:
            #     first_eos = pred_eos_positions[0].item()
            #     valid_mask[i, first_eos+1:] = False

            # Also find first EOS in labels (ground truth)
            label_pad_positions = (labels[i] == -100).nonzero(as_tuple=True)[-1]
            if len(label_pad_positions) > 0:
                last_eos = label_pad_positions[-1].item()
                preds[i, :last_eos] = torch.tensor(pad_token_id).to(preds.device)

        return preds


class CtrlQwenModelDpo(CtrlQwenModel):

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
        self.model_ref = AutoModelForCausalLM.from_pretrained(model_name).eval()
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
            input_ids=batch['input_ids_do'],
            attention_mask=batch['attention_mask_do'],
            labels=batch['target_ids_do']
        ).logits
        preferred_logprobs = self.get_log_prob(
            preferred_logits,
            batch['target_ids_do'],
            batch['target_attention_mask_do']
        )

        rejected_logits = self.model(
            input_ids=batch['input_ids_do'],
            attention_mask=batch['attention_mask_do'],
            labels=batch['target_ids_contrastive_do']
        ).logits
        rejected_logprobs = self.get_log_prob(
            rejected_logits,
            batch['target_ids_contrastive_do'],
            batch['target_attention_mask_contrastive_do']
        )

        with torch.no_grad():
            ref_preferred_logits = self.model_ref(
                input_ids=batch['input_ids_do'],
                attention_mask=batch['attention_mask_do'],
                labels=batch['target_ids_do']
            ).logits
            ref_preferred_logprobs = self.get_log_prob(
                ref_preferred_logits,
                batch['target_ids_do'],
                batch['target_attention_mask_do']
            )
            ref_rejected_logits = self.model_ref(
                input_ids=batch['input_ids_do'],
                attention_mask=batch['attention_mask_do'],
                labels=batch['target_ids_contrastive_do']
            ).logits
            ref_rejected_logprobs = self.get_log_prob(
                ref_rejected_logits,
                batch['target_ids_contrastive_do'],
                batch['target_attention_mask_contrastive_do']
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
                 on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        # rejected_rewards
        avg_rejected_rewards = loss_output[2].mean()
        self.log(f'{self.eval_log_type}_avg_rejected_rewards', avg_rejected_rewards,
                 on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        # reward margin
        avg_reward_margin = avg_chosen_rewards - avg_rejected_rewards
        self.log(f'{self.eval_log_type}_avg_reward_margin', avg_reward_margin,
                 on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

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


class CtrlQwenModelCpo(CtrlQwenModel):
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
            input_ids=batch['input_ids_do'],
            attention_mask=batch['attention_mask_do'],
            labels=batch['target_ids_do']
        )
        preferred_logits = preferred_output.logits
        preferred_loss = preferred_output.loss
        preferred_logprobs = self.get_log_prob(
            preferred_logits,
            batch['target_ids_do'],
            batch['target_attention_mask_do']
        )

        rejected_logits = self.model(
            input_ids=batch['input_ids_do'],
            attention_mask=batch['attention_mask_do'],
            labels=batch['target_ids_contrastive_do']
        ).logits
        rejected_logprobs = self.get_log_prob(
            rejected_logits,
            batch['target_ids_contrastive_do'],
            batch['target_attention_mask_contrastive_do']
        )


        return {
            'logits': preferred_logits,
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
                 on_step=True, on_epoch=True, prog_bar=True)
        # rejected_rewards
        avg_rejected_rewards = self.cpo_beta * outputs['rejected_logprobs']
        avg_rejected_rewards = avg_rejected_rewards.mean().detach()
        self.log(f'{self.eval_log_type}_avg_rejected_rewards', avg_rejected_rewards,
                 on_step=True, on_epoch=True, prog_bar=True)
        # reward margin
        avg_reward_margin = avg_chosen_rewards - avg_rejected_rewards
        self.log(f'{self.eval_log_type}_avg_reward_margin', avg_reward_margin,
                 on_step=True, on_epoch=True, prog_bar=True)

        return cpo_loss
