import os
import re
import json

from tqdm import tqdm

import pandas as pd

from sacrebleu.metrics import BLEU
from comet import load_from_checkpoint


class CpoPreferenceDataGenerator:
    """
    Class for synthetic preference data generators following CPO's approach.
    """

    def __init__(self, list_sample_data_paths,
                 src_lang, tgt_lang, save_path,
                 criteria='translation_quality',
                 attribute_type='formality',
                 ref_free_eval_model=True,
                 ref_rejected=False,
                 ref_preferred=False,
                 bleu_threshold=False,):

        # Languages
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.attribute_type = attribute_type

        # Load the samples from the provided data paths
        self.load_samples(list_sample_data_paths)

        # Criteria for generating synthetic data
        self.criteria = criteria

        self.bleu_threshold = bleu_threshold
        if self.bleu_threshold:
            self.bleu = BLEU()

        self.ref_rejected = ref_rejected
        self.ref_preferred = ref_preferred

        # Reference-free evaluation model (if any)
        if ref_free_eval_model:
            # self.eval_model = load_from_checkpoint(download_model('Unbabel/XCOMET-XXL'))
            self.eval_model = load_from_checkpoint('./pretrained_lms/Unbabel-wmt23-cometkiwi-da-xxl/checkpoints/model.ckpt')

        # Synthetic samples
        self.synthetic_samples = []

        # Save path
        self.save_path = save_path


    def load_samples(self, list_data_paths):
        """
        Load the samples from the provided data paths.
        Then merge dataframes using the index column to align the samples.
        """

        dict_samples = {}
        self.model_names = []
        for num, data_path in enumerate(list_data_paths):
            model_name = data_path.split("/")[-1].replace(".csv", "")
            self.model_names.append(model_name)
            for i, row in pd.read_csv(data_path, encoding='utf-8').iterrows():
                if row['index'] not in dict_samples:
                    dict_samples[row['index']] = {}
                dict_samples[row['index']][f'{model_name}_{num}'] = row['prediction']

        # Take one of the dataframes as a base
        base_df = pd.read_csv(list_data_paths[0], encoding='utf-8')
        base_df['source'] = base_df.apply(lambda x: x['source'].replace(x['attribute_label'], '').strip(), axis=1)
        # drop the 'predition' column
        base_df = base_df.drop(columns=['prediction'])
        # Add a 'samples' column with the samples from the dict
        base_df['samples'] = base_df['index'].apply(lambda x: dict_samples[x])
        self.samples_df = base_df

    def generate(self):
        """
        Generate synthetic preference data.
        This method should be implemented by subclasses.
        """

        if self.criteria == 'translation_quality':
            self.generate_translation_quality()
        elif self.criteria == 'translation_and_formality':
            self.generate_translation_and_formality(ref_rejected=self.ref_rejected, ref_preferred=self.ref_preferred)
        else:
            raise('Error criteria')

    def generate_translation_quality(self):
        """
        Generate synthetic preference data based on translation quality.
        """

        self.data_stats = {'preferred': {}, 'rejected': {}}

        for model_name in self.model_names:
            self.data_stats['preferred'][model_name] = 0
            self.data_stats['rejected'][model_name] = 0
        self.data_stats['preferred']['reference'] = 0
        self.data_stats['rejected']['reference'] = 0

        data = []
        labels = []
        all_metadata = []
        # Iterate over the samples and create synthetic samples
        for index, row in tqdm(self.samples_df.iterrows(), total=len(self.samples_df)):

            # Form a batch for evaluation with each sample
            source = row['source']
            reference = row['reference']
            samples = row['samples']
            attribute = row['attribute_label']

            samples['reference'] = reference

            for key, sample in samples.items():
                data.append({'src': source, 'mt': sample})
                labels.append(key)
                all_metadata.append({
                    'row_index': index,
                    'label': key,
                    'source': source,
                    'reference': reference,
                    'attribute': attribute,
                    'samples': samples
                })

        # Single prediction call
        model_output = self.eval_model.predict(data, batch_size=1, gpus=1)


        # Process results
        current_row = 0
        for index, row in self.samples_df.iterrows():
            source = row['source']
            reference = row['reference']
            samples = row['samples']
            attribute = row['attribute_label']
            samples['reference'] = reference
            num_samples = len(samples)
            row_scores = model_output.scores[current_row:current_row+num_samples]
            row_labels = [all_metadata[i]['label'] for i in range(current_row, current_row + num_samples)]
            row_indexes = [all_metadata[i]['row_index'] for i in range(current_row, current_row + num_samples)]

            # Ensure all the index values are the same
            if len(set(row_indexes)) > 1:
                raise ValueError('All index values are not the same')

            # Sort the samples based on the model output
            sorted_samples = sorted(zip(row_labels, row_scores), key=lambda x: x[1], reverse=True)

            # Best sample is preferred, last is rejected
            preferred_score = sorted_samples[0][1]
            label_preferred = sorted_samples[0][0]
            preferred = samples[label_preferred]
            self.data_stats['preferred'][label_preferred] += 1
            rejected_score = sorted_samples[-1][1]
            label_rejected = sorted_samples[-1][0]
            rejected = samples[label_rejected]
            self.data_stats['rejected'][label_rejected] += 1

            # Create a synthetic sample
            self.synthetic_samples.append((source, preferred, rejected, attribute))

            current_row += num_samples


    def generate_translation_and_formality(self, ref_rejected=False, ref_preferred=False):
        """
        Generate synthetic preference data based on translation quality.
        """

        self.data_stats = {'preferred': {}, 'rejected': {}}
        if self.attribute_type == 'formality':
            self.data_stats['preferred']['formal'] = 0
            self.data_stats['preferred']['informal'] = 0
            self.data_stats['preferred']['unknown'] = 0
            self.data_stats['rejected']['formal'] = 0
            self.data_stats['rejected']['informal'] = 0
            self.data_stats['rejected']['unknown'] = 0
        elif self.attribute_type == 'gender':
            self.data_stats['preferred']['feminine'] = 0
            self.data_stats['preferred']['masculine'] = 0
            self.data_stats['preferred']['unknown'] = 0
            self.data_stats['rejected']['feminine'] = 0
            self.data_stats['rejected']['masculine'] = 0
            self.data_stats['rejected']['unknown'] = 0

        data = []
        labels = []
        all_metadata = []
        list_num_samples = []
        # Iterate over the samples and create synthetic samples
        for index, row in tqdm(self.samples_df.iterrows(), total=len(self.samples_df)):

            # Form a batch for evaluation with each sample
            source = row['source']
            reference = row['reference']
            contrastive = row['contrastive']
            reference_annotated = row['reference_annotated']
            contrastive_annotated = row['contrastive_annotated']
            samples = row['samples']
            samples = {f'pred_{i}': val for i, (_, val) in enumerate(samples.items())}
            attribute = row['attribute_label']
            if self.attribute_type == 'formality':
                contrastive_attribute = 'informal' if attribute == 'formal' else 'formal'
            elif self.attribute_type == 'gender':
                contrastive_attribute = 'masculine' if attribute == 'feminine' else 'feminine'

            if ref_rejected:
                samples['contrastive'] = contrastive
            else:
                # Find contrastive samples
                contrastive_samples = self.samples_df.loc[
                    (self.samples_df['source'] == source) &
                    (self.samples_df['attribute_label'] == contrastive_attribute)]['samples'].values[0]
                contrstv_samples = {f'cntrst_pred_{i}': val for i, (_, val) in enumerate(contrastive_samples.items())}
                samples['reference'] = reference
                samples['contrastive'] = contrastive
                samples = samples | contrstv_samples

            count_valid_samples = 0
            for key, sample in samples.items():

                if not isinstance(sample, str):
                    continue
                count_valid_samples += 1

                pred_attribute = self.predicted_attribute(sample, reference_annotated, contrastive_annotated,
                                                          attribute, contrastive_attribute)

                data.append({'src': source, 'mt': sample})
                labels.append(key)
                all_metadata.append({
                    'row_index': index,
                    'label': key,
                    'source': source,
                    'reference': reference,
                    'attribute': attribute,
                    'pred_attribute': pred_attribute,
                    'samples': samples
                })

            list_num_samples.append(count_valid_samples)

        # Single prediction call
        model_output = self.eval_model.predict(data, batch_size=1, gpus=1)

        # Process results
        current_row = 0
        for index, row in self.samples_df.iterrows():
            source = row['source']
            reference = row['reference']
            contrastive = row['contrastive']
            samples = row['samples']
            samples = {f'pred_{i}': val for i, (_, val) in enumerate(samples.items())}

            attribute = row['attribute_label']
            if self.attribute_type == 'formality':
                contrastive_attribute = 'informal' if attribute == 'formal' else 'formal'
            elif self.attribute_type == 'gender':
                contrastive_attribute = 'masculine' if attribute == 'feminine' else 'feminine'

            if ref_rejected:
                samples['contrastive'] = contrastive
            else:
                # Find contrastive samples
                contrastive_samples = self.samples_df.loc[
                    (self.samples_df['source'] == source) &
                    (self.samples_df['attribute_label'] == contrastive_attribute)]['samples'].values[0]
                contrstv_samples = {f'cntrst_pred_{i}': val for i, (_, val) in enumerate(contrastive_samples.items())}
                samples['reference'] = reference
                samples['contrastive'] = contrastive
                samples = samples | contrstv_samples

            num_samples = list_num_samples[index]
            row_scores = model_output.scores[current_row:current_row+num_samples]
            row_labels = [all_metadata[i]['label'] for i in range(current_row, current_row + num_samples)]
            row_indexes = [all_metadata[i]['row_index'] for i in range(current_row, current_row + num_samples)]
            row_pred_attribute = [all_metadata[i]['pred_attribute'] for i in range(current_row, current_row + num_samples)]

            # Ensure all the index values are the same
            if len(set(row_indexes)) > 1:
                raise ValueError('All index values are not the same')

            # Sort the samples based on the model output
            sorted_samples = sorted(zip(row_labels, row_scores, row_pred_attribute), key=lambda x: x[1], reverse=True)

            # Best sample is preferred, last is rejected
            sorted_samples_pref = []
            for label, score, row_pred_attribute in sorted_samples:
                if row_pred_attribute == attribute:
                    if ref_preferred:
                        if label == 'reference':
                            sorted_samples_pref.append((label, score, row_pred_attribute))
                    else:
                        sorted_samples_pref.append((label, score, row_pred_attribute))
            sorted_samples_rej = []
            for label, score, row_pred_attribute in sorted_samples:
                if row_pred_attribute == contrastive_attribute:
                    if ref_rejected:
                        if label == 'contrastive':
                            sorted_samples_rej.append((label, score, row_pred_attribute))
                    else:
                        sorted_samples_rej.append((label, score, row_pred_attribute))

            if len(sorted_samples_pref):
                preferred_score = sorted_samples_pref[0][1]
                label_preferred = sorted_samples_pref[0][0]
                preferred = samples[label_preferred]
                self.data_stats['preferred'][sorted_samples_pref[0][2]] += 1
            else:
                preferred = None

            if len(sorted_samples_rej):
                rejected_score = sorted_samples_rej[-1][1]
                label_rejected = sorted_samples_rej[-1][0]
                rejected = samples[label_rejected]
                self.data_stats['rejected'][sorted_samples_rej[0][2]] += 1
            else:
                rejected = None

            # Create a synthetic sample
            if preferred is not None and rejected is not None:
                if preferred_score > rejected_score:
                    if self.bleu_threshold:
                        bleu_score = self.bleu.sentence_score(preferred, [rejected]).score
                        if bleu_score < 30:
                            continue
                    self.synthetic_samples.append((source, preferred, rejected, attribute))

            current_row += num_samples


    def predicted_attribute(self, pred, ref, contrastive_ref, attribute, contrastive_attribute):

        matches = re.findall(r'\[F\](.*?)\[\/F\]', ref)
        not_to_match = re.findall(r'\[F\](.*?)\[\/F\]', contrastive_ref)
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
            if self.tgt_lang in ['ja', 'zh']:
                pattern = re.escape(match)
            else:
                pattern = r'\b' + re.escape(match) + r'\b'
            if re.search(pattern, pred):
                matched_label = True
                break
        matched_contrastive_label = False
        for match in not_to_match:
            if self.tgt_lang in ['ja', 'zh']:
                pattern = re.escape(match)
            else:
                pattern = r'\b' + re.escape(match) + r'\b'
            if re.search(pattern, pred):
                matched_contrastive_label = True
                break

        if matched_label and not matched_contrastive_label:
            return attribute
        if matched_contrastive_label and not matched_label:
            return contrastive_attribute

        return 'Unknown'


    def save_synthetic_data(self):
        """
        Save the generated synthetic data to a file.
        """

        df = pd.DataFrame(self.synthetic_samples, columns=['source', 'preferred', 'rejected', 'attribute'])

        # If directory does not exist, create it
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        df.to_csv(f'{self.save_path}/synthetic_pref_data.csv', index=False, encoding='utf-8')

        # Save the model names
        with open(f'{self.save_path}/model_names.txt', 'w', encoding='utf-8') as f:
            for model_name in self.model_names:
                f.write(f"{model_name}\n")
            f.write('reference\n')  # Add reference as a model name

        # Save winner stats
        with open(f'{self.save_path}/data_stats.json', 'w', encoding='utf-8') as f:
            json.dump(self.data_stats, f, indent=4, ensure_ascii=False)
