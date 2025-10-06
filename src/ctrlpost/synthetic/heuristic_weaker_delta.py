import random

from transformers import AutoTokenizer

from src.ctrlpost.synthetic import BaseSyntheticPreferenceDataGenerator

random.seed(42)  # Fix the random seed for reproducibility


class HeuristicWeakerDelta(BaseSyntheticPreferenceDataGenerator):
    """
    A heuristic generates synthetic preference data by artificially "weakening"
    the quality of the rejected sample by randomly dropping tokens from the sentence.
    """

    def __init__(self, train_data_src, train_data_att1, train_data_att2,
                 src_lang, tgt_lang,
                 attr1_name, attr2_name, save_path,
                 tokenizer):
        super().__init__(train_data_src, train_data_att1, train_data_att2, src_lang, tgt_lang, attr1_name, attr2_name,
                         save_path)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.num_drop_tokens = 1

    def generate(self, num_samples_generate: int):
        """
        Generate synthetic preference data by weakening the rejected sample.
        """

        self.synthetic_samples = []
        while len(self.synthetic_samples) < num_samples_generate:
            for triplet in self.gold_triplets:
                src, att1, att2 = triplet
                # Randomly drop tokens from the rejected sample
                # First consider att2 as the rejected sample
                tokenized_att2 = self.tokenizer.tokenize(att2)
                # Randomly drop self.num_drop_tokens tokens from att2, but keep the same order of the remaining tokens
                if len(tokenized_att2) <= self.num_drop_tokens:
                    continue
                tokens_to_drop = random.sample(tokenized_att2, self.num_drop_tokens)
                for token in tokens_to_drop:
                    tokenized_att2.remove(token)
                weakened_att2 = self.tokenizer.convert_tokens_to_string(tokenized_att2)
                self.synthetic_samples.append((src, att1, weakened_att2, self.attr1_name))
                if len(self.synthetic_samples) >= num_samples_generate:
                    break
                # Then consider att1 as the rejected sample
                tokenized_att1 = self.tokenizer.tokenize(att1)
                if len(tokenized_att1) <= self.num_drop_tokens:
                    continue
                tokens_to_drop = random.sample(tokenized_att1, self.num_drop_tokens)
                for token in tokens_to_drop:
                    tokenized_att1.remove(token)
                weakened_att1 = self.tokenizer.convert_tokens_to_string(tokenized_att1)
                self.synthetic_samples.append((src, att2, weakened_att1, self.attr2_name))
