import mlflow
from mlflow.tracking import MlflowClient


def get_csv_artifacts_from_experiment(experiment_name_or_id, dict_conditions=None, only_eval_true=False):
    """
    Get all CSV artifacts from all runs in an MLflow experiment.

    Args:
        experiment_name_or_id: Either experiment name (str) or experiment ID (str)

    Returns:
        dict: Dictionary mapping run_id -> list of CSV artifact paths
    """
    client = MlflowClient()

    # Get experiment by name or ID
    if isinstance(experiment_name_or_id, str) and not experiment_name_or_id.isdigit():
        experiment = client.get_experiment_by_name(experiment_name_or_id)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name_or_id}' not found")
        experiment_id = experiment.experiment_id
    else:
        experiment_id = str(experiment_name_or_id)

    # Get all runs from the experiment
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        run_view_type=mlflow.entities.ViewType.ALL  # Include active, deleted runs
    )
    csv_artifacts = {}

    for run in runs:
        run_id = run.info.run_id

        if run.info.lifecycle_stage == 'deleted':
            continue

        # temp = run.data.params.get('temperature')
        # if float(temp) < temp_threshold:
        #     continue

        if only_eval_true:
            if not run.data.tags.get('eval') == 'true':
                print(f"  Skipping run {run_id} (not an eval run)")
                continue

        if dict_conditions is not None:
            if not conditions_satisfied(run, dict_conditions):
                continue

        try:
            # List all artifacts for this run
            artifacts = client.list_artifacts(run_id)

            # Filter for CSV files (including nested directories)
            csv_files = []
            _collect_csv_artifacts(client, run_id, artifacts, csv_files)

            if csv_files:
                csv_artifacts[run_id] = csv_files
                # print(f"  Found {len(csv_files)} CSV files: {csv_files}")
            else:
                print(f"  No CSV files found")

        except Exception as e:
            print(f"  Error accessing artifacts for run {run_id}: {e}")

    return csv_artifacts


def conditions_satisfied(run, dict_conditions):
    """
    Check if the run satisfies the given conditions.

    Args:
        run: MLflow run object
        dict_conditions: Dictionary of conditions to check

    Returns:
        bool: True if all conditions are satisfied, False otherwise
    """

    for key, value in dict_conditions.items():
        print(key)
        print(value)
        run_val = run.data.params.get(key)
        if not isinstance(value[1], str):
            run_val = type(value[1])(run_val)
        if value[0] == '=':
            if run_val != value[1]:
                return False
        elif value[0] == '!=':
            if run_val == value[1]:
                return False
        elif value[0] == '<':
            if run_val >= value[1]:
                return False
        elif value[0] == '>':
            if run_val <= value[1]:
                return False
        elif value[0] == '<=':
            if run_val > value[1]:
                return False
        elif value[0] == '>=':
            if run_val < value[1]:
                return False
    return True


def _collect_csv_artifacts(client, run_id, artifacts, csv_files, path_prefix=""):
    """Recursively collect CSV artifacts from nested directories."""
    for artifact in artifacts:
        if artifact.is_dir:
            # Recursively search in subdirectories
            nested_artifacts = client.list_artifacts(run_id, artifact.path)
            _collect_csv_artifacts(client, run_id, nested_artifacts, csv_files, artifact.path + "/")
        else:
            # Check if file ends with .csv
            if artifact.path.endswith('.csv'):
                csv_files.append(artifact.path)


def download_csv_artifacts(csv_paths):
    """
    Download specific CSV artifacts from a run.

    Args:
        run_id: MLflow run ID
        csv_paths: List of CSV artifact paths to download
        download_dir: Local directory to save files
    """
    client = MlflowClient()

    downloaded_files = {}

    for run_id, artifact_path in csv_paths.items():
        try:
            # Download the artifact
            local_path = client.download_artifacts(run_id, artifact_path[0])
            # Get run name
            run_name = client.get_run(run_id).data.tags.get('mlflow.runName', 'unknown_run')
            downloaded_files[run_name] = local_path
        except Exception as e:
            print(f"Error downloading")

    return downloaded_files