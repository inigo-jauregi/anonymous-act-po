import os
import pandas as pd


class BaseSyntheticPreferenceDataGenerator:
    """
    Base class for synthetic preference data generators.
    """

    def __init__(self, train_data_src, train_data_att1, train_data_att2,
                 src_lang, tgt_lang,
                 attr1_name, attr2_name,
                 save_path):

        # Languages
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # Attribute names
        self.attr1_name = attr1_name
        self.attr2_name = attr2_name

        self.load_gold_data(train_data_src, train_data_att1, train_data_att2)

        # Synthetic samples
        self.synthetic_samples = []

        # Save path
        self.save_path = save_path


    def load_gold_data(self, train_data_src, train_data_att1, train_data_att2):
        """
        From the train data path, load the gold data.
        This includes the whole triple of (src, tgt_att1, tgt_att2) for each sample.
        """

        # Load source texts
        with open(train_data_src, 'r', encoding='utf-8') as f:
            # Read all lines into list
            source_texts = f.readlines()

        # Load target attribute 1 texts
        with open(train_data_att1, 'r', encoding='utf-8') as f:
            # Read all lines into list
            target_texts_att1 = f.readlines()

        # Load target attribute 2 texts
        with open(train_data_att2, 'r', encoding='utf-8') as f:
            # Read all lines into list
            target_texts_att2 = f.readlines()

        self.gold_triplets = []
        for src, att1, att2 in zip(source_texts, target_texts_att1, target_texts_att2):
            # Strip newlines and spaces
            src = src.strip()
            att1 = att1.strip()
            att2 = att2.strip()
            # Append the triplet to the list
            self.gold_triplets.append((src, att1, att2))

    def generate(self, num_samples_generate: int):
        """
        Generate synthetic preference data.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def save_synthetic_data(self):
        """
        Save the generated synthetic data to a file.
        """

        df = pd.DataFrame(self.synthetic_samples, columns=['source', 'preferred', 'rejected', 'attribute'])

        # If directory does not exist, create it
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        df.to_csv(f'{self.save_path}/synthetic_pref_data.csv', index=False, encoding='utf-8')
