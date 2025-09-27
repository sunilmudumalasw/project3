"""
Unit test for the model 
Author: Sunil Mudumala
Date: 26 Sep 2025
"""

import pytest
import sys
import logging
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

from ml.data import process_data
from ml.model import inference


MODEL_PATH = Path(__name__).resolve().parent / "model"
DATA_PATH = Path(__name__).resolve().parent / "data"
DATA_FILE = DATA_PATH / "census.csv"
ENCODER_FILE = MODEL_PATH / "encoder.pkl"
LB_FILE = MODEL_PATH / "label_binarizer.pkl"
MODEL_FILE = MODEL_PATH / "trained_model.pkl"

logger = logging.getLogger(__name__)

cat_features = ["workclass", "education", "marital_status", "occupation",
                "relationship", "race", "sex", "native_country"]


@pytest.fixture(scope="session")
def df():
    """
    Fixture to load dataset

    Returns
    -------
    Pandas dataframe
    """
    df = pd.read_csv(DATA_FILE)

    yield df


def test_imported_data():
    """
    Test that the data file is imported
    """
    assert DATA_FILE.exists(), f"File not found -> {DATA_FILE}"


def test_data_shape(df):
    """
    Test that the imported data file is not empty
    """
    assert df.shape[0] > 0, f"Number of rows must be greater than 0:{df.shape[0]}"
    assert df.shape[1] > 0, f"Number of column must be greater than 0:{df.shape[1]}"


def test_features_in_data(df):
    """
    Test that the features are present in the data
    """

    assert sorted(set(df.columns).intersection(cat_features)) == sorted(cat_features)

    logger.info("Features needed are present in the dataframe")


def test_model_files():
    """
    Test that the model files are generated and saved
    """

    assert ENCODER_FILE.exists(), f"Encoder file not found -> {ENCODER_FILE}"
    assert LB_FILE.exists(), f"Label binarizer file not found -> {LB_FILE}"
    assert MODEL_FILE.exists(), f"Model file not found -> {MODEL_FILE}"

    logger.info("Model, Encoder and Label Binarizer files exist")


def test_inference_func(df):
    """
    Verify that the model files and inference function is valid
    """
    model = pickle.load(open(MODEL_FILE, "rb"))
    encoder = pickle.load(open(ENCODER_FILE, "rb"))
    lb = pickle.load(open(LB_FILE, "rb"))
    logger.info("Model file loaded")

    train, test = train_test_split(df,
                                   test_size=0.20,
                                   stratify=df["salary"])

    X_test, y_test, _encoder, _lb = process_data(test,
                                                 categorical_features=cat_features,
                                                 label="salary",
                                                 training=False,
                                                 encoder=encoder,
                                                 lb=lb)
    logger.info("Test data loaded correctly: ENCODER and LB files are valid")
    try:
        _ = inference(model, X_test)
        logger.info("Inference performed successfully")
    except Exception as err:
        logging.error("Inference function failed!")
        raise err


if __name__ == "__main__":
    sys.exit(pytest.main(["-vv", str(Path.cwd())]))