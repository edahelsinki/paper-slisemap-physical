"""
    This script trains the Slisemap models for the datasets.
"""

import sys
from pathlib import Path

import joblib
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from slisemap import Slisemap
from slisemap.local_models import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.jets import load_jets
from data.qm9 import load_qm9


def train_qm9(n=10_000, lasso=0.01, ridge=0.001):
    path = (
        Path(__file__).parent.parent
        / "models"
        / f"qm9_nn_{n//1000}k_x_35_{lasso}_{ridge}.sm"
    )
    if path.exists():
        return
    print("Loading QM9")
    # Loading, scaling, and splitting of QM9 with interpretable features
    X, y = load_qm9(model="nn")
    comb = pd.concat((X.reset_index(drop=True), y.reset_index(drop=True)), axis=1)
    mask = comb.drop_duplicates().index
    del comb
    X = X.iloc[mask, :]
    y = y.iloc[mask, :]
    scale_X = StandardScaler()
    scale_y = StandardScaler()
    (
        X_tr,
        X_te,
        y_tr,
        y_te,
        smiles_tr,
        smiles_te,
    ) = train_test_split(
        scale_X.fit_transform(X),
        scale_y.fit_transform(y),
        y.index,
        train_size=n,
        random_state=42,
    )
    # Create Slisemap object with metadata
    sm = Slisemap(X_tr, y_tr, lasso=lasso, ridge=ridge, random_state=42)
    sm.metadata.set_rows(smiles_tr)
    sm.metadata.set_variables(X.columns)
    sm.metadata.set_targets(y.columns)
    sm.metadata.set_scale_X(scale_X.mean_, scale_X.scale_)
    sm.metadata.set_scale_Y(scale_y.mean_, scale_y.scale_)
    # Train Slisemap
    print("Training QM9")
    sm.optimise(verbose=1)
    sm.save(path)


def train_jets(n=10_000, lasso=0.0001, ridge=0.0001):
    path = (
        Path(__file__).parent.parent
        / "models"
        / f"jets_rf_{n//1000}k_{lasso}_{ridge}.sm"
    )
    if path.exists():
        return
    print("Loading Jets")
    # Load and prepare the data
    X, y = load_jets(model="rf")
    scaler = StandardScaler()
    X_tr, X_te, y_tr, y_te = train_test_split(
        scaler.fit_transform(X),
        y,
        train_size=n,
        random_state=42,
        stratify=load_jets()[1],
    )
    # Create Slisemap object with metadata
    sm = Slisemap(
        X_tr,
        y_tr,
        local_model=LogisticRegression,
        lasso=lasso,
        ridge=ridge,
        random_state=42,
    )
    sm.metadata.set_variables(X_tr.columns)
    sm.metadata.set_targets(list(y_tr.columns))
    sm.metadata.set_scale_X(scaler.mean_, scaler.scale_)
    # Train Slisemap
    print("Training Jets")
    sm.optimise(verbose=1)
    sm.save(path)


def train_geckoq(lasso=1e-3, ridge=1e-3):
    path = (
        Path(__file__).parent.parent / "models" / f"geckoq_32k_expf_{lasso}_{ridge}.sm"
    )
    if path.exists():
        return
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        print("GPU not available, reverting to CPU!", flush=True)
        device = torch.device("cpu")

    hyperparams = {"device": device, "lasso": lasso, "ridge": ridge}
    print("Loading GeckoQ")
    work_directory = Path(__file__).parent.parent / "GeckoQ/"
    data = pd.read_pickle(work_directory / "Gecko_Q_expF_published_zero_mean.pkl.gz")
    variables = [x for x in data.columns if x != "pSat_Pa"]
    X = torch.tensor(data[variables].to_numpy(), dtype=torch.float32)
    Y = torch.atleast_2d(
        torch.tensor(data["pSat_Pa"].to_numpy(), dtype=torch.float32)
    ).T
    sm = Slisemap(X, Y, intercept=True, **hyperparams)
    sm = Slisemap(X, Y, lasso=lasso, ridge=ridge, random_state=42)
    sm.metadata.set_variables(variables)
    print("Training GeckoQ")
    sm.optimise(verbose=1)
    sm.save(path)


if __name__ == "__main__":
    train_qm9()
    train_jets()
    train_geckoq()
