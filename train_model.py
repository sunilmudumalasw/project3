"""
Script to train machine learning model.
Author: Sunil Mudumala
Date: 23 Sep 2025
"""
import logging
import pandas as pd
import pickle

from pathlib import Path
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference, compute_slices


MODEL_PATH = Path(__name__).resolve().parent / "model"
DATA_PATH = Path(__name__).resolve().parent / "data"
DATA_FILE = DATA_PATH / "census_clean.csv"

logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train_modell(datafile=DATA_FILE) -> None:
    logger.info(f"Reading data: {datafile}")
    data = pd.read_csv(datafile)

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    logger.info("Data Split")
    train, test = train_test_split(data, test_size=0.20, stratify=data["salary"])

    cat_features = ["workclass", "education", "marital_status", "occupation",
                    "relationship", "race", "sex", "native_country"]

    logger.info("Processing training data")
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    logger.info("Processing test data")
    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, label="salary", training=False,
        encoder=encoder, lb=lb
    )

    logger.info("Training model")
    model = train_model(X_train, y_train)

    logger.info(f"Training model complete! Saving it to {MODEL_PATH}")
    pickle.dump(model, open(MODEL_PATH / "trained_model.pkl", "wb"))
    pickle.dump(encoder, open(MODEL_PATH / "encoder.pkl", "wb"))
    pickle.dump(lb, open(MODEL_PATH / "label_binarizer.pkl", "wb"))

    logger.info("Model evaluation")
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    logging.info(f"\n\tprecision-> {precision}\n\trecall->{recall}\n\tfbeta->{fbeta}")

    logger.info("Model metrics by slice")

    for feature in cat_features:
        performance = compute_slices(test, feature, y_test, preds)
        logging.info(f"***\nPerformance on {feature} slice -> \n{performance}\n***")

if __name__ == "__main__":
    train_modell()
