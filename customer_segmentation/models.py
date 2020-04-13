import pandas as pd

from collections import namedtuple
from typing import List, Tuple, Union
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from joblib import dump, load

from .constants import PATH_MODELS, RANDOM_STATE


Features = namedtuple('Features', 'X_train X_test X_valid')
Labels = namedtuple('Labels', 'y_train y_test y_valid')
Metrics = namedtuple('Metrics', 'ACC AUC')


def cat_features_fillna(df: pd.DataFrame, cat_features: List[str]) -> pd.DataFrame:
    """Fills NA values for each column in `cat_features` for
    `df` dataframe
    """
    df_copy = df.copy()

    for cat in cat_features:
        try:
            df_copy[cat] = df_copy[cat].cat.add_categories('UNKNOWN').fillna('UNKNOWN')
        except AttributeError:
            # AttributeError is raised when the type is object instead of category
            df_copy[cat] = df_copy[cat].fillna('UNKNOWN')

    return df_copy


def preprocessing_baseline(df: pd.DataFrame,
                           cat_features: List[str],
                           target: str) -> Tuple[Features, Labels]:
    """Makes preprocessing tasks for baseline model with
    data in `df` dataframe

    Target feature must be provided in `target` arg
    """
    X = df.drop(columns=target)
    y = df[target]

    X_filled = cat_features_fillna(X, cat_features=cat_features)

    X_train, X_test_and_valid, y_train, y_test_and_valid = train_test_split(
        X_filled, y, test_size=.3, random_state=RANDOM_STATE
    )

    X_test, X_valid, y_test, y_valid = train_test_split(
        X_test_and_valid, y_test_and_valid, test_size=.5, random_state=RANDOM_STATE
    )

    return (Features(X_train, X_test, X_valid),
            Labels(y_train, y_test, y_valid))


def compute_metrics(model: CatBoostClassifier, X: pd.DataFrame, y: pd.Series) -> Metrics:
    predict = model.predict(X)
    predict_proba = model.predict_proba(X)[:, 1]

    acc = accuracy_score(y, predict)
    auc = roc_auc_score(y, predict_proba)

    return Metrics(ACC=acc, AUC=auc)


def show_metrics_baseline(model: CatBoostClassifier, features: Features, labels: Labels) -> None:
    """Giving `model`, `features` and `labels` show accuracy and AUC
    for training, testing and validation data

    Model passed in argument `model` has to be already fitted
    """
    acc_train, auc_train = compute_metrics(model, X=features.X_train, y=labels.y_train)
    acc_test, auc_test = compute_metrics(model, X=features.X_test, y=labels.y_test)
    acc_valid, auc_valid = compute_metrics(model, X=features.X_valid, y=labels.y_valid)

    print(f'Accuracy Train: {acc_train}')
    print(f'Accuracy Test: {acc_test}')
    print(f'Accuracy Valid: {acc_valid}')

    print(f'AUC Train: {auc_train}')
    print(f'AUC Test: {auc_test}')
    print(f'AUC Valid: {auc_valid}')


def target_stats_by_feature(df: pd.DataFrame, feature: str,
                            target: str, fillna_value: Union[str, float] = None) -> pd.DataFrame:
    """Computes the mean and the volume of `target` for each value of `feature`
    """
    df_copy = (
        df.loc[:, [feature, target]].fillna(fillna_value) if fillna_value
        else df.loc[:, [feature, target]]
    )

    df_grouped = (
        df_copy
            .groupby(feature)[target]
            .agg(['mean', 'count'])
            .reset_index()
    )

    df_grouped.columns = [feature, f'{target}_mean', f'{target}_count']

    return df_grouped.sort_values(by=f'{target}_mean', ascending=False)


def save_catboost_model(catboost_model: CatBoostClassifier, model_name: str, pool_data: Pool) -> None:
    """Saves model `catboost_model` to `PATH_MODELS` with the name
    passed in `model_name`

    `pool_data` contains `Pool` object with features and lkabels used
    to fit the model and its categorical features
    """
    catboost_model.save_model(str(PATH_MODELS / model_name), pool=pool_data)


def load_catboost_model(model_name: str) -> CatBoostClassifier:
    """Reads `model_name` from `PATH_MODELS` and returns
    the fitted catboost model
    """
    test_model_from_file = CatBoostClassifier()

    test_model_from_file.load_model(str(PATH_MODELS / model_name))

    return test_model_from_file


def save_pipeline(pipeline: Pipeline, model_name: str) -> None:
    """Saves model `pipeline` to `PATH_MODELS` with the name
    passed in `model_name`
    """
    dump(pipeline, PATH_MODELS / model_name)


def load_pipeline(model_name: str) -> CatBoostClassifier:
    """Reads `model_name` from `PATH_MODELS` and returns
    the fitted catboost model
    """
    return load(PATH_MODELS / model_name)
