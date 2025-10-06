'''
CtrlBLOOM baseline class. Class for baseline BLOOM multi-lingual LLM
'''

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BloomTokenizerFast,
    BloomForCausalLM,
    get_linear_schedule_with_warmup
)
from typing import Dict, Any, Optional

from src.ctrlpost.models.base_model import BaseTranslationModel


class CtrlBloomModel(BaseTranslationModel):
    def __init__(
            self,
            model_name: str,
            **kwargs
    ):
        """
        Bloom model for translation.

        Args:
            model_name (str): Hugging Face model name/path
            **kwargs: Additional configuration parameters
        """

        tokenizer = BloomTokenizerFast.from_pretrained(model_name)
        model = BloomForCausalLM.from_pretrained(model_name,
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
            # 'temperature': 0.3,
            'do_sample': False,
            # 'top_p': 0.9,
            'eos_token_id': tokenizer.eos_token_id,
            # 'early_stopping': True,
            # 'no_repeat_ngram_size': 3,
            # 'repetition_penalty': 1.2,
            # 'length_penalty': 0.8
        }
        super().__init__(**kwargs)

        if kwargs['lora_finetuning']:
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


        preds = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            # TODO: Potentially add return_dict_in_generate=True and output_scores=True (for CRPO calculations)
            **self.extra_config.get('generation_config', {})
        )

        return preds
