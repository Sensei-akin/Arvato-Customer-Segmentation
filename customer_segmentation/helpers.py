import pandas as pd
import numpy as np
import random

from typing import Dict, Set, Any
from kaggle.api.kaggle_api_extended import KaggleApi
from joblib import dump, load

from .constants import PATH_DATA, PATH_FILE_ATTRIBUTES, PATH_SUBMISSIONS, SEP, NA_VALUES, RANDOM_STATE, PATH_OBJECTS


def read_attributes() -> pd.DataFrame:
    """Reads attributes file provided by Arvato
    """
    df_attributes = pd.read_excel(PATH_FILE_ATTRIBUTES, header=1, usecols=['Attribute', 'Description', 'Meaning'])

    df_attributes.columns = map(str.lower, df_attributes.columns)

    return df_attributes


def dtypes_from_attributes(df: pd.DataFrame) -> Dict[str, str]:
    """Builds dtype dictionary from dataframe `df` saving
    a lot of memory with 'category' approach vs 'object' one
    """
    df_copy = df.dropna().copy()

    df_copy['is_numeric'] = df_copy['meaning'].str.startswith('numeric value')

    dict_attributes = (df_copy[['attribute', 'is_numeric']]
                       .set_index('attribute')
                       .to_dict()
                       .get('is_numeric'))

    return {attribute: float if is_numeric else 'category'
            for attribute, is_numeric in dict_attributes.items()}


def read_demographic_data(filename: str, sample_ratio: float = 1.0) -> pd.DataFrame:
    """Reads `sample_ratio` [0, 1] sample of demographic data from
    `filename` located in `PATH_DATA` path

    Uses correct dtypes for saving memory
    """
    df_attributes = read_attributes()
    dtype = dtypes_from_attributes(df_attributes)

    def skiprows_sample(index: int) -> bool:
        """Inner function to read the data in a sampled way"""
        return index > 0 and random.random() > sample_ratio

    # RANDOM_STATE is necessary for easy reproducibility of sampled data
    random.seed(RANDOM_STATE)

    return pd.read_csv(PATH_DATA / filename,
                       sep=SEP,
                       dtype=dtype,
                       na_values=NA_VALUES,
                       skiprows=None if sample_ratio == 1 else skiprows_sample)


def constant_columns(df: pd.DataFrame) -> Set[str]:
    """Gives the constants columns of `df` dataframe
    """
    df_dropped_constants = df.loc[:, (df != df.iloc[0]).any()]

    return set(df.columns).difference(df_dropped_constants.columns)


def nullity_pct(df: pd.DataFrame) -> pd.Series:
    """Returns the percentage of nulls for each column
    """
    return df.isnull().mean()


def kaggle_submission(column_lnr: pd.Series, y_pred: np.array,
                      submission_filename: str, submission_message: str) -> None:
    """Submits and saves submission data provided in `column_lrt` and
    `y_pred`
    """
    filepath = PATH_SUBMISSIONS / f'{submission_filename}.csv'
    df_kaggle_submission = pd.DataFrame(dict(LNR=column_lnr, RESPONSE=y_pred))

    df_kaggle_submission.to_csv(filepath, index=False)

    kaggle_api = KaggleApi()
    kaggle_api.authenticate()

    print(kaggle_api.competition_submit(filepath,
                                        message=submission_message,
                                        competition='udacity-arvato-identify-customers'))


def serialize_object_dump(object_: Any, filename: str) -> None:
    """Dumps `object` in `PATH_OBJECTS / filename`
    """
    dump(object_, PATH_OBJECTS / filename)


def serialize_object_load(filename: str) -> Any:
    """Loads serialized object in `PATH_OBJECTS / filename`
    """
    return load(PATH_OBJECTS / filename)
