import argparse
from src.ctrlpost.synthetic import CpoPreferenceDataGenerator
from src.ctrlpost.utils import get_csv_artifacts_from_experiment, download_csv_artifacts

def main(args):

    conditions = None


    list_artifacts = get_csv_artifacts_from_experiment(args.experiment_name,
                                                       dict_conditions=conditions)
    dict_paths = download_csv_artifacts(list_artifacts)
    list_paths = [val for _, val in dict_paths.items()]

    ref_pref = True if args.fixed == 'pref' else False
    ref_rej = True if args.fixed == 'rej' else False

    data_generator = CpoPreferenceDataGenerator(
        list_sample_data_paths=list_paths,
        src_lang=args.src,
        tgt_lang=args.tgt,
        attribute_type='gender',
        save_path=f'./synthetic_data/mt_geneval/{args.experiment_name}/synthetic_data_{args.fixed}',
        criteria='translation_and_formality',
        ref_free_eval_model=True,
        ref_preferred=ref_pref,
        ref_rejected=ref_rej,
        bleu_threshold=False,
    )

    # Generate synthetic samples

    data_generator.generate()

    # Save the synthetic samples
    data_generator.save_synthetic_data()

    del data_generator


parser = argparse.ArgumentParser(description='Train a translation model')
parser.add_argument('--experiment-name', type=str, required=True,
                    help='Path to the training data')
parser.add_argument('--src', type=str, required=True,
                    help='Path to the training data')
parser.add_argument('--tgt', type=str, required=True,
                    help='Path to the training data')
parser.add_argument('--fixed', type=str, required=True,
                    help='Path to the validation data')
if __name__ == '__main__':
    my_args = parser.parse_args()
    main(my_args)
