import mlflow
import mlflow.sklearn

from mlflow.exceptions import MlflowException

from .constants import PATH_MLFLOW_TRACKING


def new_experiment(name: str) -> str:
    """Creates an experiment in `PATH_MLFLOW_TRACKING` and
    return the experiment id

    If the experiment already exists, return its id
    """
    mlflow.set_tracking_uri(str(PATH_MLFLOW_TRACKING))
    try:
        return mlflow.create_experiment(name)
    except MlflowException:
        return mlflow.get_experiment_by_name(name).experiment_id


def new_run(experiment_id, run_name, tags, params, metrics, model, model_name) -> str:
    """Creates a new run in the experiment with id `experiment_id`
    and return the run id
    """
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        mlflow.set_tags(tags)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, model_name)

        return run.info.run_uuid
