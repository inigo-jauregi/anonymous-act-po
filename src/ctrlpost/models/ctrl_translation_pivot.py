'''
Ctrl Translation using a pivot intermediate language
'''

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)
from typing import Dict, Optional

from src.ctrlpost.models.base_model import BaseTranslationModel
from src.ctrlpost.utils.language_mappings import lang_token_mapping


class CtrlTranslationPivotModel(BaseTranslationModel):
    def __init__(
            self,
            model_name: str,
            model_name_2: str,
            pivot_lng: str,
            train_model_1: bool = False,
            train_model_2: bool = False,
            **kwargs
    ):
        """
        Base PyTorch Lightning model for translation and APE tasks.

        Args:
            model_name (str): Hugging Face model name/path
            model_name_2 (str): Second model name/path (optional)
            pivot_lng (str): The intermediate pivot language (generally English)
            train_model_1 (bool): Whether to train the first model
            train_model_2 (bool): Whether to train the second model
            **kwargs: Additional configuration parameters
        """
        super().__init__( **kwargs)

        # Load pretrained model and tokenizer
        pivot_lang_tkn = lang_token_mapping(pivot_lng, 'NLLB')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                       tgt_lang=pivot_lang_tkn)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        if model_name_2:
            tgt_lang_tkn = lang_token_mapping(kwargs['tgt_lng'], 'NLLB')
            self.tokenizer_2 = AutoTokenizer.from_pretrained(model_name_2,
                                                             tgt_lang=tgt_lang_tkn)
            self.model_2 = AutoModelForSeq2SeqLM.from_pretrained(model_name_2)
        else:
            self.tokenizer_2 = None
            self.model_2 = None
        self.train_model_2 = train_model_2
        self.pivot_lng = pivot_lng

        self.train_model_1 = train_model_1
        self.train_model_2 = train_model_2

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

        if self.train_model_2:
            # Use the second model for training
            return self.model_2(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['target_ids']
            )

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

        bos_pivot_token_nllb = lang_token_mapping(self.pivot_lng, 'NLLB')
        bos_token_nllb = lang_token_mapping(self.tgt_lng, 'NLLB')

        # Only run model 1 inference
        if self.train_model_1:
            # first translation into pivot language
            preds_1 = self.model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(bos_pivot_token_nllb),
                **self.extra_config.get('generation_config', {})
            )
            return preds_1

        if self.train_model_2:
            # first translation into pivot language
            preds_2 = self.model_2.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                forced_bos_token_id=self.tokenizer_2.convert_tokens_to_ids(bos_token_nllb),
                **self.extra_config.get('generation_config', {})
            )
            return preds_2

        # If none of the two, run the whole pipeline (both models)

        # first translation into pivot language
        preds_1 = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(bos_pivot_token_nllb),
            **self.extra_config.get('generation_config', {})
        )

        # detokenize the pivot language
        preds_1_detok = self.tokenizer.batch_decode(preds_1, skip_special_tokens=True)
        # Append the attribute label to all the samples
        if self.controllable:
            preds_1_detok = [f"{batch['attribute_labels'][i]} {sentence}" for i, sentence in enumerate(preds_1_detok)]

        if self.model_2:
            # tokenize the pivot language
            preds_1_tok = self.tokenizer_2(preds_1_detok,
                                           return_tensors='pt',
                                           max_length=512,
                                           padding=True,
                                           padding_side=batch['padding_side'][0],
                                           truncation=True).to(self.model.device)

            preds = self.model_2.generate(
                input_ids=preds_1_tok['input_ids'],
                attention_mask=preds_1_tok['attention_mask'],
                forced_bos_token_id=self.tokenizer_2.convert_tokens_to_ids(bos_token_nllb),
                **self.extra_config.get('generation_config', {})
            )
        else:
            # tokenize the pivot language
            preds_1_tok = self.tokenizer(preds_1_detok,
                                           return_tensors='pt',
                                           max_length=512,
                                           padding=True,
                                           padding_side=batch['padding_side'][0],
                                           truncation=True).to(self.model.device)

            preds = self.model.generate(
                input_ids=preds_1_tok['input_ids'],
                attention_mask=preds_1_tok['attention_mask'],
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(bos_token_nllb),
                **self.extra_config.get('generation_config', {})
            )

        return preds

    def overwrite_arguments(self, args):

        self.train_model_1 = args.train_model_1
        self.train_model_2 = args.train_model_2

        # Decide tgt language according to which model is being trained
        if self.train_model_1:
            self.tgt_lng = 'en'
        elif self.train_model_2:
            self.src_lng = 'en'
