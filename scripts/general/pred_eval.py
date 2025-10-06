import argparse
import os

import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from comet import download_model, load_from_checkpoint

from src.ctrlpost.utils import evaluation
from src.ctrlpost.utils import get_csv_artifacts_from_experiment, download_csv_artifacts


def fix_cocoa_predictions(df_preds, cocoa_dataset_path, src_lang, tgt_lang):

    # Load dataset
    # Read un-annotated source text
    with open(f"{cocoa_dataset_path}.{src_lang}-{tgt_lang}.{src_lang}", 'r', encoding='utf-8') as f:
        # Read all lines into list
        source_texts = f.readlines()
    # Read formal un-annotated target text
    with open(f"{cocoa_dataset_path}.{src_lang}-{tgt_lang}.formal.{tgt_lang}", 'r', encoding='utf-8') as f:
        target_texts_formal = f.readlines()
    # Read formal annotated target text
    with open(f"{cocoa_dataset_path}.{src_lang}-{tgt_lang}.formal.annotated.{tgt_lang}", 'r', encoding='utf-8') as f:
        target_texts_formal_annotated = f.readlines()
    # Read informal un-annotated target text
    with open(f"{cocoa_dataset_path}.{src_lang}-{tgt_lang}.informal.{tgt_lang}", 'r', encoding='utf-8') as f:
        target_texts_informal = f.readlines()
    # Read informal annotated target text
    with open(f"{cocoa_dataset_path}.{src_lang}-{tgt_lang}.informal.annotated.{tgt_lang}", 'r', encoding='utf-8') as f:
        target_texts_informal_annotated = f.readlines()

    # Combine into single list
    source_texts = source_texts + source_texts
    target_texts = target_texts_formal + target_texts_informal
    # target_texts_annotated = target_texts_formal_annotated + target_texts_informal_annotated
    # formality_labels = ['formal'] * len(target_texts_formal) + ['informal'] * len(target_texts_informal)
    target_texts_contrastive = target_texts_informal + target_texts_formal
    target_texts_annotated_contrastive = target_texts_informal_annotated + target_texts_formal_annotated

    # Rename columns
    df_preds = df_preds.rename(columns={
        'source': 'prompt_source',
        'reference': 'reference_annotated',
        'formality_label': 'attribute_label'
    })

    # Add new columns
    df_preds['source'] = source_texts
    df_preds['reference'] = target_texts
    df_preds['contrastive'] = target_texts_contrastive
    df_preds['contrastive_annotated'] = target_texts_annotated_contrastive

    return df_preds


def prepare_predictions(df_preds, include_comet=False, include_ref_free_comet=False, duplicate_src_inv_label=False):

    if duplicate_src_inv_label:
        df_preds_copy = df_preds[:]
        for i, row in df_preds_copy.iterrows():
            gt_sentence = row['reference']
            attribute_label = row['attribute_label']
            df_oppo_sample = df_preds_copy[(df_preds_copy['reference'] == gt_sentence) & (df_preds_copy['attribute_label'] != attribute_label)]
            if len(df_oppo_sample) != 1:
                raise ValueError(f"More than one opposite sample found for {i}th sample. Expected 1.")
            oppo_sample_src = df_oppo_sample.iloc[0]['source']
            df_preds_copy.loc[i, 'source'] = oppo_sample_src

        df_preds = pd.concat([df_preds, df_preds_copy], ignore_index=True)

    prep_dict = {
        'prediction_list': [str(pred) for pred in df_preds['prediction'].tolist()],
        'src_raw_list': df_preds['source'].tolist(),
        'src_list': df_preds['prompt_source'].tolist(),
        'gt_list': df_preds['reference'].tolist(),
        'gt_annotated_list': df_preds['reference_annotated'].tolist(),
        'contrastive_list': df_preds['contrastive'].tolist(),
        'gt_annotated_list_contrastive': df_preds['contrastive_annotated'].tolist(),
        'attribute_label_list': df_preds['attribute_label'].tolist()
    }

    if include_comet:
        # Form comet data structure
        data_comet = []
        for i, row in df_preds.iterrows():
            data_comet.append({'src': row['source'],
                               'mt': str(row['prediction']),
                               'ref': row['reference']})
        prep_dict['data_comet'] = data_comet

    if include_ref_free_comet:
        # Form ref-free comet data structure
        data_ref_free_comet = []
        for i, row in df_preds.iterrows():
            data_ref_free_comet.append({'src': row['source'],
                                        'mt': str(row['prediction'])})
        prep_dict['data_ref_free_comet'] = data_ref_free_comet

    return prep_dict


def is_mlflow_experiment_name(name):
    client = MlflowClient()
    # Get experiment by name or ID
    if isinstance(name, str):
        experiment = client.get_experiment_by_name(name)
        if experiment is None:
            return False
        return True
    return False


def main(args):
    """
    Main function to run the evaluation script.
    """

    # Load COMET model
    if args.include_comet:
        comet_model = load_from_checkpoint(download_model(args.include_comet))
        include_comet = True
    else:
        comet_model = None
        include_comet = False

    # Load reference-free COMET model
    if args.include_ref_free_comet:
        ref_free_comet_model = load_from_checkpoint(args.include_ref_free_comet)
        include_ref_free_comet = True
    else:
        ref_free_comet_model = None
        include_ref_free_comet = False

    # Check if the args.preds_file is a valid mlflow experiment name
    exp_name = None
    if is_mlflow_experiment_name(args.preds_file):
        conditions = {
            # 'objective': ('=', 'dpo'),
            'tag': ('eval', 'true')
        }
        # Extract runs from the mlflow experiment
        list_artifacts = get_csv_artifacts_from_experiment(args.preds_file,
                                                           dict_conditions=conditions,
                                                           only_eval_true=args.only_eval_true)
        dict_paths = download_csv_artifacts(list_artifacts)
        exp_name = args.preds_file

    elif args.preds_file.startswith('file:///'):
        # Load artifact from mlflow
        mlflow_preds_path = mlflow.artifacts.download_artifacts(artifact_uri=args.preds_file)
        dict_paths = {'model': mlflow_preds_path}
    else:
        dict_paths = {'model': args.preds_file}


    df_output = pd.DataFrame()
    for model_name, csv_path in dict_paths.items():
        df_preds = pd.read_csv(csv_path)

        # (optional) Fix old predictions if a cocoa dataset is provided
        if args.fix_cocoa_preds:
            df_preds = fix_cocoa_predictions(df_preds, args.fix_cocoa_preds, src_lang=args.src_lng, tgt_lang=args.tgt_lng)

        # Prepare the predictions for evaluation
        prep_all_predictions = prepare_predictions(df_preds, include_comet, include_ref_free_comet)

        # Run evaluation
        dict_results = evaluation(prep_all_predictions, args.tgt_lng,
                                  comet_model=comet_model,
                                  ref_free_comet=ref_free_comet_model,
                                  gpt_judge=args.gpt_judge)
        # Print results
        print('Evaluation Results:')
        for key, value in dict_results.items():
            print(f'\t{key}: {value}')

        # Add dict results as a row in the dataframe, keys as columns and values as values, include an additional column for the model name
        dict_results['model_name'] = model_name
        df_output = pd.concat([df_output, pd.DataFrame([dict_results])], ignore_index=True)

    # Write to a csv file (output directory)
    # 'output'
    # Create output directory if it does not exist
    output_dir = './output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Write to a csv file
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    if exp_name:
        outfile = f'{output_dir}/evaluation_results_{exp_name}_{timestamp}.csv'
    else:
        outfile = f'{output_dir}/evaluation_results_{timestamp}.csv'
    df_output.to_csv(outfile, index=False)


parser = argparse.ArgumentParser(description='Train a translation model')
parser.add_argument('--preds-file', type=str, required=True,
                    help='Path to the file containing predictions for evaluation')
parser.add_argument('--src-lng', type=str, required=True,
                    help='Source language for the evaluation')
parser.add_argument('--tgt-lng', type=str, required=True,
                    help='Target language for the evaluation')
parser.add_argument('--include-comet', type=str, default=None,
                    help='Include COMET evaluation metric calculation.'
                         'If not provided, it will not be included.')
parser.add_argument("--include-ref-free-comet", type=str, default=None,
                    help='Include reference-free COMET evaluation metric calculation.'
                         'If not provided, it will not be included.')
parser.add_argument('--gpt-judge', action='store_true',
                    help='Include GPT as a judge for formality accuracy metric calculation.'
                         'If not provided, it will not be included.')
parser.add_argument('--only-eval-true', action="store_true",
                    help='Only evaluate the predictionswith the tag "eval=true" in the mlflow experiment.'),
parser.add_argument('--fix-cocoa-preds', type=str, default=None,
                    help='Path to the file containing path to the cocoa dataset in order to fix the old predictions'
                         'and include all the relevant inputs and outputs required for evaluation.'
                         'Example -> ./data/CoCoA_MT/test/en-de/formality-control.test.all')
parser.add_argument('--duplicate-src-inv-label', action="store_true",
                    help='Duplicate the source sentence with the opposite label for evaluation.'
                         'This is useful for evaluating the contrastive evaluation metric.')
if __name__ == '__main__':
    my_args = parser.parse_args()
    main(my_args)
