'''
Dataloaders for the COCOA-MT dataset.
https://aclanthology.org/2022.findings-naacl.47.pdf
'''
import random
import re

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer
from typing import List, Optional, Union
import pandas as pd

from src.ctrlpost.utils.language_mappings import lang_token_mapping


class CocoaMtDataset(Dataset):
    def __init__(
            self,
            path_to_dataset: str,
            src_lang: str,
            tgt_lang: str,
            tokenizer: Union[str, AutoTokenizer],
            model_type: str = 'encoder-decoder',
            max_length: int = 512,
            padding: str = 'longest',
            padding_side: str = 'right',
            truncation: bool = True,
            num_samples: Optional[int] = None,
            contrastive_dataset: bool = False,
            prompt: str = None,
            in_context_learning: bool = False,
            in_context_sample_pool: Dataset = None,
            in_context_num_samples: int = 1,
            dpo_training: bool = False,
            add_synthetic_pref_data: List[str] = None,
            only_synthetic: bool = False,
            remove_neutral_train_data: bool = False,
    ):
        """
        Custom Dataset for translation and APE tasks.

        Args:
            source_texts (List[str]): List of source language texts
            target_texts (List[str]): List of target language texts
            tokenizer (Union[str, AutoTokenizer]): Tokenizer or tokenizer name
            model_type (str): Type of model to use ('encoder-decoder' or 'decoder-only'). Defaults to 'encoder-decoder'.
            max_length (int, optional): Maximum sequence length. Defaults to 512.
            padding (str, optional): Padding strategy. Defaults to 'max_length'.
            padding_side (str, optional): Padding side. Defaults to 'right'.
            truncation (bool, optional): Whether to truncate sequences. Defaults to True.
            contrastive_dataset (bool, optional): Whether to use contrastive dataset. Defaults to False.
            prompt (str, optional): Prompt to prepend to source texts. Defaults to None.
            in_context_learning (bool, optional): Whether to use in-context learning. Defaults to False.
            in_context_sample_pool (Dataset, optional): Dataset to sample in-context examples from.
            in_context_num_samples (int, optional): Number of in-context samples to use. Defaults to 1.
            dpo_training (bool, optional): Whether to use DPO training. Defaults to False.
        """

        tgt_lang_tkn = lang_token_mapping(tgt_lang, 'NLLB')

        # Ensure tokenizer is loaded
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer,
                                                           # token=True,
                                                           # src_lang=f"eng_Latn",
                                                           tgt_lang=tgt_lang_tkn)
        else:
            self.tokenizer = tokenizer

        # Store language codes
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # Preference tuning training flag
        self.dpo_training = dpo_training

        # Dataset parameters
        self.model_type = model_type
        self.max_length = max_length
        self.padding = padding
        self.padding_side = padding_side
        self.truncation = truncation

        # Load dataset
        # Read un-annotated source text
        with open(f"{path_to_dataset}.{src_lang}-{tgt_lang}.{src_lang}", 'r', encoding='utf-8') as f:
            # Read all lines into list
            source_texts = f.readlines()
            if num_samples is not None:
                source_texts = source_texts[:num_samples]
        # Read formal un-annotated target text
        with open(f"{path_to_dataset}.{src_lang}-{tgt_lang}.formal.{tgt_lang}", 'r', encoding='utf-8') as f:
            target_texts_formal = f.readlines()
            if num_samples is not None:
                target_texts_formal = target_texts_formal[:num_samples]
        # Read formal annotated target text
        with open(f"{path_to_dataset}.{src_lang}-{tgt_lang}.formal.annotated.{tgt_lang}", 'r', encoding='utf-8') as f:
            target_texts_formal_annotated = f.readlines()
            if num_samples is not None:
                target_texts_formal_annotated = target_texts_formal_annotated[:num_samples]
        # Read informal un-annotated target text
        with open(f"{path_to_dataset}.{src_lang}-{tgt_lang}.informal.{tgt_lang}", 'r', encoding='utf-8') as f:
            target_texts_informal = f.readlines()
            if num_samples is not None:
                target_texts_informal = target_texts_informal[:num_samples]
        # Read informal annotated target text
        with open(f"{path_to_dataset}.{src_lang}-{tgt_lang}.informal.annotated.{tgt_lang}", 'r', encoding='utf-8') as f:
            target_texts_informal_annotated = f.readlines()
            if num_samples is not None:
                target_texts_informal_annotated = target_texts_informal_annotated[:num_samples]

        # Remove neutral training data if specified
        if remove_neutral_train_data:
            source_texts_rmvd = []
            target_texts_formal_rmvd = []
            target_texts_formal_annotated_rmvd = []
            target_texts_informal_rmvd = []
            target_texts_informal_annotated_rmvd = []
            for src, tgt_formal, tgt_formal_annotated, tgt_informal, tgt_informal_annotated in \
                    zip(source_texts, target_texts_formal, target_texts_formal_annotated,
                        target_texts_informal, target_texts_informal_annotated):
                if tgt_formal == tgt_informal:
                    continue
                source_texts_rmvd.append(src)
                target_texts_formal_rmvd.append(tgt_formal)
                target_texts_formal_annotated_rmvd.append(tgt_formal_annotated)
                target_texts_informal_rmvd.append(tgt_informal)
                target_texts_informal_annotated_rmvd.append(tgt_informal_annotated)
            # Update the lists with the removed neutral data
            source_texts = source_texts_rmvd
            target_texts_formal = target_texts_formal_rmvd
            target_texts_formal_annotated = target_texts_formal_annotated_rmvd
            target_texts_informal = target_texts_informal_rmvd
            target_texts_informal_annotated = target_texts_informal_annotated_rmvd

        # Combine into single list
        source_texts = source_texts + source_texts
        target_texts = target_texts_formal + target_texts_informal
        target_texts_annotated = target_texts_formal_annotated + target_texts_informal_annotated
        formality_labels = ['formal'] * len(target_texts_formal) + ['informal'] * len(target_texts_informal)

        if add_synthetic_pref_data:
            syn_src_texts, syn_tgt_texts, syn_contrastive_texts, syn_formality_labels = \
                self.add_synthetic_preference_data(add_synthetic_pref_data)
            if only_synthetic:
                source_texts = syn_src_texts
                target_texts = syn_tgt_texts
                target_texts_annotated = len(syn_src_texts) * ['']
                formality_labels = syn_formality_labels
            else:
                source_texts += syn_src_texts
                target_texts += syn_tgt_texts
                target_texts_annotated += len(syn_src_texts) * ['']
                formality_labels += syn_formality_labels

        # Consider prepending the formality label to the input text
        # ['<INPUT_SRC>',  '<SRC_LANG>', '<TGT_LANG>', '<FORMALITY>']
        raw_source_texts = []
        if prompt:
            prompted_source_texts = []
        # source_texts = [f"[{formality}] {text}" for formality, text in zip(formality_labels, source_texts)]
            for formality, text in zip(formality_labels, source_texts):

                # Copy prompt to avoid modifying the original
                prompt_text = prompt

                if in_context_learning:
                    in_context_prompt = self.retrieve_in_context_samples(in_context_sample_pool, prompt,
                                                                         in_context_num_samples)
                    prompt_text = in_context_prompt + prompt_text

                # Remove <OUTPUT_TGT> and subsequent tokens from prompt
                prompt_text = self.remove_target_and_after(prompt_text)
                prompt_text = self.fill_prompt_template(prompt_text, input_src=text, formality=formality)
                prompted_source_texts.append(prompt_text)
                raw_source_texts.append(text)
            source_texts = prompted_source_texts
        else:
            source_texts = [f"{text.strip()}" for formality, text in zip(formality_labels, source_texts)]

        self.encodings, self.encodings_do = self.tokenize(
            source_texts,
            target_texts
        )

        self.input_texts = source_texts
        self.source_texts = raw_source_texts
        self.target_texts = target_texts
        self.formality_labels = formality_labels
        self.target_texts_annotated = target_texts_annotated
        self.target_texts_contrastive = None
        self.target_texts_annotated_contrastive = None
        if contrastive_dataset:
            if add_synthetic_pref_data:
                if only_synthetic:
                    self.target_texts_contrastive = syn_contrastive_texts
                    self.target_texts_annotated_contrastive = syn_contrastive_texts
                else:
                    self.target_texts_contrastive = target_texts_informal + target_texts_formal + syn_contrastive_texts
                    self.target_texts_annotated_contrastive = target_texts_informal_annotated + target_texts_formal_annotated + syn_contrastive_texts
            else:
                self.target_texts_contrastive = target_texts_informal + target_texts_formal
                self.target_texts_annotated_contrastive = target_texts_informal_annotated + target_texts_formal_annotated
            self.contrastive_dataset = True
        if dpo_training:
            # For DTO we need to tokenize the contrastive example as well
            self.encodings_contrastive, self.encodings_contrastive_do = self.tokenize(
                source_texts,
                self.target_texts_contrastive
            )
        else:
            self.encodings_contrastive = None
            self.encodings_contrastive_do = None

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):

        if self.encodings is None:
            sample_dict = {
                'input_text': self.input_texts[idx],
                'raw_source_text': self.source_texts[idx],
                'target_text': self.target_texts[idx],
                'attribute_labels': self.formality_labels[idx],
                'target_annotated_text': self.target_texts_annotated[idx],
                'padding_side': self.padding_side,
            }
        else:
            target_attention_mask = (self.encodings['labels'] != self.tokenizer.pad_token_id).long()
            target_lengths = target_attention_mask.sum(dim=1)
            sample_dict = {
                'raw_source_text': self.source_texts[idx],
                'input_ids': self.encodings['input_ids'][idx],
                'attention_mask': self.encodings['attention_mask'][idx],
                'target_ids': self.encodings['labels'][idx],
                'target_attention_mask': target_attention_mask[idx],
                'target_lengths': target_lengths[idx],
                'attribute_labels': self.formality_labels[idx],
                'target_annotated_text': self.target_texts_annotated[idx],
                'padding_side': self.padding_side,
            }

        if self.encodings_do:
            target_attention_mask_do = (self.encodings_do['labels'] != self.tokenizer.pad_token_id).long()
            target_lengths_do = target_attention_mask_do.sum(dim=1)
            sample_dict['input_ids_do'] = self.encodings_do['input_ids'][idx]
            sample_dict['attention_mask_do'] = self.encodings_do['attention_mask'][idx]
            sample_dict['target_ids_do'] = self.encodings_do['labels'][idx]
            sample_dict['source_lengths_do'] = self.encodings_do['source_lengths'][idx]
            sample_dict['target_attention_mask_do'] = target_attention_mask_do[idx]
            sample_dict['target_lengths_do'] = target_lengths_do[idx]

        if self.target_texts_annotated_contrastive:
            sample_dict['target_text_contrastive'] = self.target_texts_contrastive[idx]
            sample_dict['target_annotated_text_contrastive'] = self.target_texts_annotated_contrastive[idx]

        if self.dpo_training:
            sample_dict['target_ids_contrastive'] = self.encodings_contrastive['labels'][idx]
            target_constrastive_mask = (self.encodings_contrastive['labels'] != self.tokenizer.pad_token_id).long()
            target_constrastive_lengths = target_constrastive_mask.sum(dim=1)
            sample_dict['target_attention_mask_contrastive'] = target_constrastive_mask[idx]
            sample_dict['target_lengths_contrastive'] = target_constrastive_lengths[idx]

        if self.encodings_contrastive_do:
            sample_dict['input_ids_contrastive_do'] = self.encodings_contrastive_do['input_ids'][idx]
            sample_dict['attention_mask_contrastive_do'] = self.encodings_contrastive_do['attention_mask'][idx]
            sample_dict['target_ids_contrastive_do'] = self.encodings_contrastive_do['labels'][idx]
            sample_dict['source_lengths_contrastive_do'] = self.encodings_contrastive_do['source_lengths'][idx]
            target_contrastive_mask_do = (self.encodings_contrastive_do['labels'] != self.tokenizer.pad_token_id).long()
            target_contrastive_lengths_do = target_contrastive_mask_do.sum(dim=1)
            sample_dict['target_attention_mask_contrastive_do'] = target_contrastive_mask_do[idx]
            sample_dict['target_lengths_contrastive_do'] = target_contrastive_lengths_do[idx]


        return sample_dict

    def tokenize(self, source_texts, target_texts):

        if self.tokenizer is not None:
            encodings = self.tokenizer(
                source_texts,
                text_target=target_texts,
                max_length=self.max_length,
                padding=self.padding,
                padding_side=self.padding_side,
                truncation=self.truncation,
                return_tensors='pt',
            )
            if self.model_type == 'decoder-only':
                encodings_do = self._tokenize_for_decoder_only(
                    source_texts,
                    target_texts,
                    max_length=self.max_length,
                    padding="left",
                    truncation=self.truncation
                )
            else:
                encodings_do = None
        else:
            encodings = None
            encodings_do = None

        return encodings, encodings_do

    def retrieve_in_context_samples(self, dataset, prompt, num_samples=1):

        samples = []
        # num_samples = 1
        for _ in range(num_samples):
            # Formal sample
            while(True):
                sample_num = random.randint(0, len(dataset.formality_labels)-1)
                # Check if annotation in target_text_sample_anotated
                if ('[F]' in dataset.target_texts_annotated[sample_num] and
                        '[/F]' in dataset.target_texts_annotated[sample_num]):
                    break
            source_text_sample = dataset.source_texts[sample_num]
            target_text_sample = dataset.target_texts[sample_num]
            target_text_sample_annotated = dataset.target_texts_annotated[sample_num]
            formality_label = dataset.formality_labels[sample_num]
            sample = self.fill_prompt_template(prompt, source_text_sample, target_text_sample,
                                                      target_text_sample_annotated, formality_label)
            sample += '\n'
            samples.append(sample)
            # Check if contrastive example available
            if dataset.contrastive_dataset:
                target_text_sample_annotated_contrastive = dataset.target_texts_annotated_contrastive[sample_num]
                target_text_sample_contrastive = (
                    target_text_sample_annotated_contrastive.replace('[F]', '').replace('[/F]', ''))
                opposite_formality_label = 'informal' if formality_label == 'formal' else 'formal'
                contrastive_sample = self.fill_prompt_template(prompt, source_text_sample,
                                                               target_text_sample_contrastive,
                                                                target_text_sample_annotated_contrastive,
                                                               opposite_formality_label)
                contrastive_sample += '\n'
                samples.append(contrastive_sample)

        # Join the samples into a single string
        all_samples = ''.join(samples)

        return all_samples

    @staticmethod
    def remove_target_and_after(text):
        index = text.find("<OUTPUT_TGT>")
        if index != -1:
            return text[:index]
        return text


    def fill_prompt_template(self, prompt, input_src=None, output_tgt=None,
                             output_tgt_annotated=None, formality=None):

        prompt_text = prompt

        if '<INPUT_SRC>' in prompt:
            prompt_text = prompt_text.replace('<INPUT_SRC>', input_src.strip())
        if '<OUTPUT_TGT>' in prompt:
            prompt_text = prompt_text.replace('<OUTPUT_TGT>', output_tgt.strip())
        if '<SRC_LANG>' in prompt:
            prompt_src_lang = lang_token_mapping(self.src_lang, 'prompt')
            prompt_text = prompt_text.replace('<SRC_LANG>', prompt_src_lang)
        if '<TGT_LANG>' in prompt:
            prompt_tgt_lang = lang_token_mapping(self.tgt_lang, 'prompt')
            prompt_text = prompt_text.replace('<TGT_LANG>', prompt_tgt_lang)
        if '<FORMALITY>' in prompt:
            prompt_text = prompt_text.replace('<FORMALITY>', formality)
        if '<FORMALITY_TOKENS>' in prompt and output_tgt_annotated:
            # Identiy the formality tokens with regex
            matches = re.findall(r'\[F\](.*?)\[\/F\]', output_tgt_annotated)
            matches = [f"'{match.strip()}'" for match in matches]
            # Join the matches with comma and space
            formality_tokens = ', '.join(matches)
            prompt_text = prompt_text.replace('<FORMALITY_TOKENS>', formality_tokens)

        return prompt_text

    def _tokenize_for_decoder_only(self, source_texts, target_texts, max_length, padding, truncation):
        """
        Tokenize source and target texts for decoder-only training.
        Concatenates source and target with proper masking.
        """
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []
        batch_source_lengths = []

        for source, target in zip(source_texts, target_texts):
            # Tokenize source and target separately to get lengths
            source_tokens = self.tokenizer(source.strip(), add_special_tokens=False)['input_ids']
            target_tokens = self.tokenizer(target.strip() + '}', add_special_tokens=False)['input_ids']

            # Create combined sequence: [source] [EOS] [target] [EOS]
            combined_ids = (
                    source_tokens +
                    target_tokens
            )

            # Truncate if necessary
            if truncation and len(combined_ids) > max_length:
                combined_ids = combined_ids[:max_length]

            # Create labels - mask source portion
            labels = combined_ids.copy()
            source_length = len(source_tokens)
            # Mask source tokens
            for i in range(source_length):
                if i < len(labels):
                    labels[i] = -100

            # Create attention mask
            attention_mask = [1] * len(combined_ids)

            batch_input_ids.append(combined_ids)
            batch_labels.append(labels)
            batch_attention_mask.append(attention_mask)
            batch_source_lengths.append(source_length)

        # Pad sequences
        if padding is not None:
            padded_input_ids = []
            padded_labels = []
            padded_attention_mask = []

            for i in range(len(batch_input_ids)):
                current_length = len(batch_input_ids[i])
                padding_length = max_length - current_length

                if padding_length > 0:
                    # Pad sequences
                    if padding == 'right':
                        padded_input_ids.append(
                            batch_input_ids[i] + [self.tokenizer.pad_token_id] * padding_length
                        )
                        padded_labels.append(
                            batch_labels[i] + [-100] * padding_length
                        )
                        padded_attention_mask.append(
                            batch_attention_mask[i] + [0] * padding_length
                        )
                    elif padding == 'left':
                        padded_input_ids.append(
                            [self.tokenizer.pad_token_id] * padding_length + batch_input_ids[i]
                        )
                        padded_labels.append(
                            [-100] * padding_length + batch_labels[i]
                        )
                        padded_attention_mask.append(
                            [0] * padding_length + batch_attention_mask[i]
                        )
                else:
                    padded_input_ids.append(batch_input_ids[i])
                    padded_labels.append(batch_labels[i])
                    padded_attention_mask.append(batch_attention_mask[i])

            batch_input_ids = padded_input_ids
            batch_labels = padded_labels
            batch_attention_mask = padded_attention_mask

        return {
            'input_ids': torch.tensor(batch_input_ids),
            'attention_mask': torch.tensor(batch_attention_mask),
            'labels': torch.tensor(batch_labels),
            'source_lengths': torch.tensor(batch_source_lengths)
        }

    def add_synthetic_preference_data(self, list_synthetic_data_paths: List[str]):
        """
        Add synthetic preference data to the dataset.
        The synthetic data should be in the format of (src, tgt_att1, tgt_att2, attr_name).
        """

        all_src_texts = []
        all_tgt_texts = []
        all_cntrstve_texts = []
        all_attr_names = []
        for path in list_synthetic_data_paths:

            # Read the synthetic data from the file
            df_synthetic = pd.read_csv(path)

            # Tokenize the input with the preferred target text
            all_src_texts += df_synthetic['source'].tolist()
            all_tgt_texts += df_synthetic['preferred'].tolist()
            all_cntrstve_texts += df_synthetic['rejected'].tolist()
            all_attr_names = df_synthetic['attribute'].tolist()

        return all_src_texts, all_tgt_texts, all_cntrstve_texts, all_attr_names


class CocoaMtDataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_data_path: str,
            val_data_path: str,
            test_data_path: str,
            src_lang: str,
            tgt_lang: str,
            tokenizer: Union[str, AutoTokenizer],
            model_type: str = 'encoder-decoder',
            batch_size: int = 8,
            max_length: int = 1024,
            padding_side: str = 'right',
            num_workers: int = 4,
            num_train_samples: Optional[int] = None,
            num_val_samples: Optional[int] = None,
            num_test_samples: Optional[int] = None,
            prompt: str = None,
            in_context_learning: bool = False,
            in_context_num_samples: int = 1,
            dpo_training: bool = False,
            add_synthetic_pref_data: List[str] = None,
            only_synthetic: bool = False,
            remove_neutral_train_data: bool = False,
    ):
        """
        Lightning DataModule for translation tasks.

        Args:
            train_data_path (str): Training data path
            val_data_path (str): Validation data path
            test_data_path (str): Test data path
            src_lang (str): Source language
            tgt_lang (str): Target language
            tokenizer (Union[str, AutoTokenizer]): Tokenizer or tokenizer name
            model_type (str, optional): Type of model to use ('encoder-decoder' or 'decoder-only'). Defaults to 'encoder-decoder'.
            batch_size (int, optional): Batch size. Defaults to 8.
            max_length (int, optional): Maximum sequence length. Defaults to 512.
            padding_side (str, optional): Padding side. Defaults to 'right'.
            num_workers (int, optional): Number of dataloader workers. Defaults to 4.
            num_train_samples (Optional[int], optional): Number of training samples. If None, use all samples.
                                                         Defaults to None.
            num_val_samples (Optional[int], optional): Number of validation samples. If None, use all samples.
                                                       Defaults to None.
            num_test_samples (Optional[int], optional): Number of test samples. If None, use all samples.
                                                        Defaults to None.
            prompt (str, optional): Prompt to prepend to source texts. Defaults to None.
            in_context_learning (bool, optional): Whether to use in-context learning. Defaults to False.
            in_context_num_samples (int, optional): Number of in-context samples to use. Defaults to 1.
            dpo_training (bool, optional): Whether to use DPO training. Defaults to False.
        """
        super().__init__()

        # Save parameters
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.test_data_path = test_data_path

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.tokenizer = tokenizer
        self.model_type = model_type
        self.batch_size = batch_size
        self.max_length = max_length
        self.padding_side = padding_side
        self.num_workers = num_workers

        # Will be set in setup method
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Optional number of samples
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples

        self.prompt = prompt
        self.in_context_learning = in_context_learning
        self.in_context_num_samples = in_context_num_samples

        self.dpo_training = dpo_training

        self.add_synthetic_pref_data = add_synthetic_pref_data
        self.only_synthetic = only_synthetic
        self.remove_neutral_train_data = remove_neutral_train_data

    def setup(self, stage: Optional[str] = None):
        """
        Set up datasets for different stages.

        Args:
            stage (Optional[str], optional): Stage of training. Defaults to None.
        """
        # Create datasets
        if stage in [None, 'fit']:
            self.train_dataset = CocoaMtDataset(
                self.train_data_path,
                self.src_lang,
                self.tgt_lang,
                self.tokenizer,
                model_type=self.model_type,
                max_length=self.max_length,
                num_samples=self.num_train_samples,
                padding_side=self.padding_side,
                prompt=self.prompt,
                dpo_training=self.dpo_training,
                add_synthetic_pref_data=self.add_synthetic_pref_data,
                only_synthetic=self.only_synthetic,
                contrastive_dataset=True,
                remove_neutral_train_data=self.remove_neutral_train_data,
            )
            self.val_dataset = CocoaMtDataset(
                self.val_data_path,
                self.src_lang,
                self.tgt_lang,
                self.tokenizer,
                model_type=self.model_type,
                max_length=self.max_length,
                padding_side=self.padding_side,
                num_samples=self.num_val_samples,
                prompt=self.prompt,
                contrastive_dataset=True,
                dpo_training=self.dpo_training
            )

        if stage in [None, 'test']:
            if self.in_context_learning:
                in_context_pool = CocoaMtDataset(
                    self.train_data_path,
                    self.src_lang,
                    self.tgt_lang,
                    self.tokenizer,
                    model_type=self.model_type,
                    max_length=self.max_length,
                    padding_side=self.padding_side,
                    num_samples=self.num_train_samples,
                    contrastive_dataset=True,
                    prompt=self.prompt
                )
            else:
                in_context_pool = None
            if self.test_data_path:
                self.test_dataset = CocoaMtDataset(
                    self.test_data_path,
                    self.src_lang,
                    self.tgt_lang,
                    self.tokenizer,
                    model_type=self.model_type,
                    max_length=self.max_length,
                    padding_side=self.padding_side,
                    num_samples=self.num_test_samples,
                    contrastive_dataset=True,
                    prompt=self.prompt,
                    in_context_learning=self.in_context_learning,
                    in_context_sample_pool=in_context_pool,
                    in_context_num_samples=self.in_context_num_samples,
                    dpo_training=self.dpo_training
                )

    def train_dataloader(self):
        """
        Create training dataloader.

        Returns:
            DataLoader: Training dataloader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

    def val_dataloader(self):
        """
        Create validation dataloader.

        Returns:
            DataLoader: Validation dataloader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

    def test_dataloader(self):
        """
        Create test dataloader.

        Returns:
            DataLoader: Test dataloader
        """
        if self.test_dataset:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )
        return None
