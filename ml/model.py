'''
Trainig model
Author: Sunil Mudumala
Date: 23 Sep 2025
'''

from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple
import numpy as np
import pandas as pd

def train_model(X_train: np.array, y_train: np.array) -> RandomForestClassifier:
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    trained_model = RandomForestClassifier()
    trained_model.fit(X_train, y_train)

    return trained_model

def compute_slices(test_df, feature: str,
                   y: np.array, preds: np.array):
    """
    Computes the model metrics for a specific feature

    Parameters
    ----------
    test_df: np.array
        test data frame
    feature : str
        feature selected to compute slices
    y : np.array
        labels
    preds : np.array
        Predicted labels

    Returns
    -------
    pandas dataframe with model metrics of the selected feature

    """

    slice_options = test_df[feature].unique().tolist()
    perf_df = pd.DataFrame(
        index=slice_options,
        columns=["feature", "n_samples", "precision", "recall", "fbeta"])

    for option in slice_options:
        slice_mask = test_df[feature] == option

        slice_y = y[slice_mask]
        slice_preds = preds[slice_mask]
        precision, recall, fbeta = compute_model_metrics(slice_y, slice_preds)

        perf_df.at[option, "feature"] = feature
        perf_df.at[option, "n_samples"] = len(slice_y)
        perf_df.at[option, "precision"] = precision
        perf_df.at[option, "recall"] = recall
        perf_df.at[option, "fbeta"] = fbeta

    return perf_df

def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model: RandomForestClassifier, X: np.array) -> np.array:
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)

    return preds
