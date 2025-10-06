'''
Methods to run evaluation of Attribute Controlled Translation (ACT) models.
'''
import re

import pandas as pd
import sacrebleu
from sacrebleu.metrics import BLEU
from pytorch_lightning.loggers import MLFlowLogger
from .gpt_as_judge import gpt_formality_evaluation


def evaluation(prep_all_predictions, tgt_lng, empty_return=False, save_preds=False,
               comet_model=None, ref_free_comet=None, gpt_judge=False, loggers=None):
    '''
    Evaluate the predictions of an ACT model.

    Code to compute the following metrics:
    - BLEU score
    - COMET score
    - Matched Accuracy (M-Acc)
    - Matched Accuracy - Strict (M-Acc-Strict)
    - Token-level Recall (T-Recall)

    :return:
    '''

    dict_scores = {}
    unique_labels = set(prep_all_predictions['attribute_label_list'])

    if empty_return:
        # Get unique attribute labels
        dict_scores['bleu'] = 0.0
        dict_scores['comet_score'] = 0.0
        for label in unique_labels:
            dict_scores[f'{label}_m_acc'] = 0.0
            dict_scores[f'{label}_strict_m_acc'] = 0.0
            dict_scores[f'{label}_token_recall'] = 0.0
        dict_scores['avg_m_acc'] = 0.0
        dict_scores['avg_strict_m_acc'] = 0.0
        dict_scores['avg_token_recall'] = 0.0
        dict_scores['m_acc_bleu'] = 0.0
        return dict_scores

    # Compute BLEU score
    bleu = sacrebleu.corpus_bleu(prep_all_predictions['prediction_list'],
                                 [prep_all_predictions['gt_list']],
                                 tokenize="ja-mecab" if tgt_lng in ['ja', 'zh'] else BLEU.TOKENIZER_DEFAULT)
    dict_scores['bleu'] = bleu.score

    # Compute COMET score
    if comet_model is not None:
        comet_score = comet_model.predict(prep_all_predictions['data_comet'],
                                          batch_size=4).system_score
        dict_scores['comet_score'] = comet_score

    if ref_free_comet is not None:
        ref_free_comet_score = ref_free_comet.predict(
            prep_all_predictions['data_ref_free_comet'],
            batch_size=4, devices=[0]
            ).system_score
        dict_scores['ref_free_comet_score'] = ref_free_comet_score

    if gpt_judge:
        # I want to prompt GPT via API to evaluate whether the prediction is formal or informal
        gpt_accuracy = gpt_formality_evaluation(prep_all_predictions)
        dict_scores['gpt_formality_accuracy'] = gpt_accuracy

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
                if tgt_lng in ['ja', 'zh']:
                    pattern = re.escape(match)
                else:
                    pattern = r'\b' + re.escape(match) + r'\b'
                if re.search(pattern, pred_text):
                    matched_label = True
                    break
            matched_contrastive_label = False
            for match in not_to_match:
                if tgt_lng in ['ja', 'zh']:
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
            for label in unique_labels:
                dict_scores[f'{label}_m_acc'] = 0.0
            dict_scores['avg_m_acc'] = 0.0

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
            for label in unique_labels:
                dict_scores[f'{label}_strict_m_acc'] = 0.0
            dict_scores['avg_strict_m_acc'] = 0.0

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
            for label in unique_labels:
                dict_scores[f'{label}_token_recall'] = 0.0
            dict_scores['avg_token_recall'] = 0.0

    # calculate average between M-Acc and BLEU
    dict_scores['m_acc_bleu'] = (dict_scores['avg_m_acc'] + dict_scores['bleu']) / 2

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
        if loggers is not None:
            for logger in loggers:
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
