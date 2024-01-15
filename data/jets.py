"""
    This scripts downloads and preprocesses the jets dataset when called directly.
    The files are placed in the same directory as this script.
"""

import pickle
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

try:
    from qm9 import exists, load, store, check_preprocess
except:
    from .qm9 import exists, load, store, check_preprocess


def preprocess_jets():
    if not exists("jets"):
        df = pd.read_csv(Path(__file__).parent / "jets.csv")
        df["particle"] = pd.Categorical(["Gluon", "Quark"])[df["isPhysUDS"]]
        del df["isPhysUDS"]
        store(df, "jets")
    rf_path = Path(__file__).parent / "jets_rf.pkl"
    if not rf_path.exists():
        X, y = load_jets(None)
        rf = RandomForestClassifier(max_leaf_nodes=50, random_state=42)
        rf.fit(X, y["particle"])
        with open(rf_path, "wb") as f:
            pickle.dump(rf, f, protocol=5)
    if not exists("jets_rf"):
        store(load_jets("rf")[1], "jets_rf")


@check_preprocess(__file__)
def load_jets(model: Literal[None, "rf"] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the particle jets dataset.

    Args:
        model: Choose an optional classifier for the targets. Defaults to None.

    Returns:
        X: Features.
        y: Targets.
    """
    df = load("jets")
    y = df["particle"].to_frame()
    del df["particle"]
    if model == "rf":
        if exists("jets_rf"):
            y = load("jets_rf")
        else:
            rf = load_model(model)
            y = rf.predict_proba(df).astype(np.float32)
            y = pd.DataFrame(y, columns=["Gluon", "Quark"])
    elif model is not None:
        raise NotImplementedError()
    return df, y


@check_preprocess(__file__)
def load_model(model: Literal["rf"] = "rf") -> RandomForestClassifier:
    """Load a particle jet classifier.

    Args:
        model: The model to load. Defaults to "rf".

    Returns:
        Sklearn classifier.
    """
    if model == "rf":
        with open(Path(__file__).parent / "jets_rf.pkl", "rb") as f:
            return pickle.load(f)
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    preprocess_jets()
