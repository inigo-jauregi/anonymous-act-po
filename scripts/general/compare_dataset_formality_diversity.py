import re
from collections import Counter
import numpy as np

import pandas as pd


def extract_annotated_tokens(sentences, annotation_tag='F'):
    """Extract all tokens within annotation tags."""
    pattern = rf'\[{annotation_tag}\](.*?)\[/{annotation_tag}\]'
    tokens = []
    for sen in sentences:
        # Formal tokens
        matches = re.findall(pattern, sen)
        tokens.extend(matches)
    return tokens


def type_token_ratio(translations, annotation_tag='F'):
    """
    Calculate Type-Token Ratio (TTR) for annotated tokens.
    TTR = unique tokens / total tokens
    Higher values indicate more diversity.
    """
    tokens = extract_annotated_tokens(translations, annotation_tag)
    if len(tokens) == 0:
        return 0.0
    return len(set(tokens)) / len(tokens)


def gini_coefficient(annotated_tokens):
    """
    Calculate Gini coefficient for token distribution.
    Lower values indicate more equal distribution (more diversity).
    Range: 0 (perfect equality) to 1 (perfect inequality).
    """
    if len(annotated_tokens) == 0:
        return 0.0

    counts = Counter(annotated_tokens)
    sorted_counts = sorted(counts.values())
    n = len(sorted_counts)

    cumsum = np.cumsum(sorted_counts)
    return (2 * np.sum((np.arange(1, n + 1)) * sorted_counts)) / (n * cumsum[-1]) - (n + 1) / n

def top_k_concentration(annotated_tokens, k=10):
    """
    Calculate percentage of annotations from top-k most frequent tokens.
    Higher values indicate lower diversity (concentration in few tokens).

    Args:
        k: Number of top tokens to consider
    """
    if len(annotated_tokens) == 0:
        return 0.0

    counts = Counter(annotated_tokens)
    top_k_counts = counts.most_common(k)
    top_k_total = sum(count for _, count in top_k_counts)

    return top_k_total / len(annotated_tokens)


def calculate_stats(tokens_formal, tokens_informal, tokes_all, annotation_tag='F'):
    """Compare Type-Token Ratio between two datasets."""

    # Formal stats
    num_formal_token_annotations = len(tokens_formal)
    num_unique_formal_tokens = len(set(tokens_formal))
    ttr_formal = num_unique_formal_tokens / num_formal_token_annotations if num_formal_token_annotations > 0 else 0.0
    # Informal stats
    num_informal_token_annotations = len(tokens_informal)
    num_unique_informal_tokens = len(set(tokens_informal))
    ttr_informal = num_unique_informal_tokens / num_informal_token_annotations if num_informal_token_annotations > 0 else 0.0

    # Total stats
    num_all_token_annotations = len(tokes_all)
    num_unique_all_tokens = len(set(tokes_all))
    ttr_all = num_unique_all_tokens / num_all_token_annotations if num_all_token_annotations > 0 else 0.0


    return {
        'formal': {
            'num_formal_token_annotations': num_formal_token_annotations,
            'num_unique_formal_tokens': num_unique_formal_tokens,
            'ttr_formal': ttr_formal
        },
        'informal': {
            'num_informal_token_annotations': num_informal_token_annotations,
            'num_unique_informal_tokens': num_unique_informal_tokens,
            'ttr_informal': ttr_informal
        },
        'all': {
            'num_all_token_annotations': num_all_token_annotations,
            'num_unique_all_tokens': num_unique_all_tokens,
            'ttr_all': ttr_all,
        }
    }


# Example usage
if __name__ == "__main__":

    # Dataset 1 is PRE-FAME-MT
    dict_formal_informal_per_lang_pfm = {}
    tokens_all_dataset_pfm = []
    for lang_pair in ['en-de', 'da-es', 'de-fr', 'pl-it', 'it-nl', 'ru-pt']:
        tgt_lang = lang_pair.split('-')[1]
        df_pfm = pd.read_csv(f'./data/PRE_FAME_MT/{lang_pair}/test_contrastive.csv')
        list_formal = df_pfm['tgt_formal_annotated'].tolist()
        list_informal = df_pfm['tgt_informal_annotated'].tolist()
        list_all = list_formal + list_informal
        # Stats per language per formal/informal
        tokens_formal = extract_annotated_tokens(list_formal)
        tokens_informal = extract_annotated_tokens(list_informal)
        tokens_all = extract_annotated_tokens(list_all)
        tokens_all_dataset_pfm.extend(tokens_all)
        dict_formal_informal_per_lang_pfm[tgt_lang] = calculate_stats(tokens_formal, tokens_informal, tokens_all)
        # Top-20 words list in descending order
        top_20_words = Counter(tokens_all).most_common(20)
        print(f'Language: {tgt_lang}')
        print(f'Top-20 words in {lang_pair}: {top_20_words}')
        df_output_words = pd.DataFrame(columns=['rank', 'PRE-FAME-MT'])
        df_output_words['rank'] = range(1, 21)
        df_output_words['PRE-FAME-MT'] = top_20_words
        df_output_words.to_csv(f'./scripts/general/pre-fame-mt-{tgt_lang}.csv', index=False)

    df_output_pfm = pd.DataFrame(columns=['group', 'num_tokens', 'num_unique_tokens', 'ttr'])
    counter = 0
    num_tokens_formal_all = 0
    num_unique_tokens_formal_all = 0
    num_tokens_informal_all = 0
    num_unique_tokens_informal_all = 0
    num_tokens_all = 0
    num_unique_tokens_all = 0
    for key, value in dict_formal_informal_per_lang_pfm.items():
        group = f'{key}-formal'
        df_output_pfm.loc[counter] = [group, value['formal']['num_formal_token_annotations'], value['formal']['num_unique_formal_tokens'], value['formal']['ttr_formal']]
        num_tokens_formal_all += value['formal']['num_formal_token_annotations']
        num_unique_tokens_formal_all += value['formal']['num_unique_formal_tokens']
        counter += 1
        group = f'{key}-informal'
        df_output_pfm.loc[counter] = [group, value['informal']['num_informal_token_annotations'], value['informal']['num_unique_informal_tokens'], value['informal']['ttr_informal']]
        num_tokens_informal_all += value['informal']['num_informal_token_annotations']
        num_unique_tokens_informal_all += value['informal']['num_unique_informal_tokens']
        counter += 1
        group = f'{key}'
        df_output_pfm.loc[counter] = [group, value['all']['num_all_token_annotations'], value['all']['num_unique_all_tokens'], value['all']['ttr_all']]
        num_tokens_all += value['all']['num_all_token_annotations']
        num_unique_tokens_all += value['all']['num_unique_all_tokens']
        counter += 1

    # Add formal
    group = 'formal'
    df_output_pfm.loc[counter] = [group, num_tokens_formal_all, num_unique_tokens_formal_all, num_unique_tokens_formal_all / num_tokens_formal_all]
    counter += 1
    # Add informal
    group = 'informal'
    df_output_pfm.loc[counter] = [group, num_tokens_informal_all, num_unique_tokens_informal_all, num_unique_tokens_informal_all / num_tokens_informal_all]
    counter += 1
    # Add total
    group = 'total'
    df_output_pfm.loc[counter] = [group, num_tokens_all, num_unique_tokens_all, num_unique_tokens_all / num_tokens_all]
    counter += 1

    print(f'Gini coefficient for PRE-FAME-MT: {gini_coefficient(tokens_all_dataset_pfm)}')
    print(f'Top-5 concentration for PRE-FAME-MT: {top_k_concentration(tokens_all_dataset_pfm, k=5)}')
    print(f'Top-10 concentration for PRE-FAME-MT: {top_k_concentration(tokens_all_dataset_pfm, k=10)}')
    print(f'Top-20 concentration for PRE-FAME-MT: {top_k_concentration(tokens_all_dataset_pfm, k=20)}')


    # Dataset 2 is CoCoA-MT
    dict_formal_informal_per_lang_cmt = {}
    tokens_all_dataset_cmt = []
    for tgt_lang in ['de', 'es', 'fr', 'it', 'nl', 'pt']:
        with open(f'./data/CoCoA_MT/test/en-{tgt_lang}/formality-control.test.all.en-{tgt_lang}.formal.annotated.{tgt_lang}', 'r') as f:
            list_formal = []
            for line in f:
                list_formal.append(line.strip())
        with open(f'./data/CoCoA_MT/test/en-{tgt_lang}/formality-control.test.all.en-{tgt_lang}.informal.annotated.{tgt_lang}', 'r') as f:
            list_informal = []
            for line in f:
                list_informal.append(line.strip())

        list_all = list_formal + list_informal
        # Stats per language per formal/informal
        tokens_formal = extract_annotated_tokens(list_formal)
        tokens_informal = extract_annotated_tokens(list_informal)
        tokens_all = extract_annotated_tokens(list_all)
        tokens_all_dataset_cmt.extend(tokens_all)
        dict_formal_informal_per_lang_cmt[tgt_lang] = calculate_stats(tokens_formal, tokens_informal, tokens_all)
        # Top-20 words list in descending order
        top_20_words = Counter(tokens_all).most_common(20)
        print(f'Language: {tgt_lang}')
        print(f'Top-20 words in {lang_pair}: {top_20_words}')
        df_output_words = pd.DataFrame(columns=['rank', 'CoCoA-MT'])
        df_output_words['rank'] = range(1, 21)
        df_output_words['CoCoA-MT'] = top_20_words
        df_output_words.to_csv(f'./scripts/general/cocoa-mt-{tgt_lang}.csv', index=False)

    df_output_cmt = pd.DataFrame(columns=['group', 'num_tokens', 'num_unique_tokens', 'ttr'])
    counter = 0
    num_tokens_formal_all = 0
    num_unique_tokens_formal_all = 0
    num_tokens_informal_all = 0
    num_unique_tokens_informal_all = 0
    num_tokens_all = 0
    num_unique_tokens_all = 0
    for key, value in dict_formal_informal_per_lang_cmt.items():
        group = f'{key}-formal'
        df_output_cmt.loc[counter] = [group, value['formal']['num_formal_token_annotations'],
                                      value['formal']['num_unique_formal_tokens'], value['formal']['ttr_formal']]
        num_tokens_formal_all += value['formal']['num_formal_token_annotations']
        num_unique_tokens_formal_all += value['formal']['num_unique_formal_tokens']
        counter += 1
        group = f'{key}-informal'
        df_output_cmt.loc[counter] = [group, value['informal']['num_informal_token_annotations'],
                                      value['informal']['num_unique_informal_tokens'],
                                      value['informal']['ttr_informal']]
        num_tokens_informal_all += value['informal']['num_informal_token_annotations']
        num_unique_tokens_informal_all += value['informal']['num_unique_informal_tokens']
        counter += 1
        group = f'{key}'
        df_output_cmt.loc[counter] = [group, value['all']['num_all_token_annotations'],
                                      value['all']['num_unique_all_tokens'], value['all']['ttr_all']]
        num_tokens_all += value['all']['num_all_token_annotations']
        num_unique_tokens_all += value['all']['num_unique_all_tokens']
        counter += 1

    # Add formal
    group = 'formal'
    df_output_cmt.loc[counter] = [group, num_tokens_formal_all, num_unique_tokens_formal_all,
                                  num_unique_tokens_formal_all / num_tokens_formal_all]
    counter += 1
    # Add informal
    group = 'informal'
    df_output_cmt.loc[counter] = [group, num_tokens_informal_all, num_unique_tokens_informal_all,
                                  num_unique_tokens_informal_all / num_tokens_informal_all]
    counter += 1
    # Add total
    group = 'total'
    df_output_cmt.loc[counter] = [group, num_tokens_all, num_unique_tokens_all,
                                  num_unique_tokens_all / num_tokens_all]
    counter += 1

    print(f'Gini coefficient for CoCoA-MT: {gini_coefficient(tokens_all_dataset_cmt)}')
    print(f'Top-5 concentration for CoCoA-MT: {top_k_concentration(tokens_all_dataset_cmt, k=5)}')
    print(f'Top-10 concentration for CoCoA-MT: {top_k_concentration(tokens_all_dataset_cmt, k=10)}')
    print(f'Top-20 concentration for CoCoA-MT: {top_k_concentration(tokens_all_dataset_cmt, k=20)}')

    # Merge both dataframes on group
    df_output_merged = pd.merge(df_output_pfm, df_output_cmt, on='group', how='inner', suffixes=('_pfm', '_cmt'))
    df_output_merged.to_csv('scripts/general/stats.csv', index=False)
