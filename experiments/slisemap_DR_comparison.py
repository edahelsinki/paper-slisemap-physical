"""This script loops over pretrained Slisemap and alternative DR models.
It is meant to be run after train_alternative_DR."""
import sys
import numpy as np
import pandas as pd
import os
from datetime import datetime

from pathlib import Path

sys.path.insert(0, "../../notes/")

from slisemap import Slisemap
from slisemap.utils import LBFGS
from slisemap.utils import global_model as _global_model

curr_path = Path(__file__)
import torch
from slisemap.metrics import coverage, fidelity, euclidean_nearest_neighbours

# set shared parameters
job_id = int(sys.argv[1]) - 1
model_folder = sys.argv[2]
random_seed = range(1618033, 1618043)[job_id]
rng = np.random.default_rng(random_seed)

# set up device
device = torch.device("cpu")


def global_model(sm: Slisemap) -> torch.Tensor:
    """Train a global model for coverage."""
    return _global_model(
        X=sm.X,
        Y=sm.Y,
        local_model=sm.local_model,
        local_loss=sm.local_loss,
        coefficients=sm.coefficients,
        lasso=sm.lasso,
        ridge=sm.ridge,
    )


def get_coverage(sm: Slisemap, q_v=0.3):
    """Wrapper for coverage."""
    global_loss = sm.local_loss(sm.local_model(sm.X, global_model(sm)), sm.Y)
    q = torch.quantile(global_loss, q_v).detach().item()
    return coverage(sm, q)


def get_coverage_nn(sm: Slisemap, q_v=0.3, k=0.2):
    """Wrapper for NN-coverage."""
    global_loss = sm.local_loss(sm.local_model(sm.X, global_model(sm)), sm.Y)
    q = torch.quantile(global_loss, q_v).detach().item()
    n = {"neighbours": euclidean_nearest_neighbours, "k": k}
    return coverage(sm, q, **n)


def get_fidelity(sm: Slisemap):
    """Compute fidelity."""
    vals = torch.diag(sm.get_L(numpy=False))
    return torch.mean(vals).detach().item()


def get_fidelity_nn(sm: Slisemap, k=0.2):
    """Wrapper for fidelity."""
    n = {"neighbours": euclidean_nearest_neighbours, "k": k}
    return fidelity(sm, **n)


# choose metrics to calculate.
metrics = {
    "total_loss": lambda sm: sm.value(),
    "fidelity": get_fidelity,
    "fidelity-nn": get_fidelity_nn,
    # 'coverage': get_coverage,
    "coverage-nn": get_coverage_nn,
}

# loop over the metrics
dicts = []
model_list = [x for x in os.listdir(model_folder) if x[-3:] == ".sm"]
model_list = [x for x in model_list if str(random_seed) in x]
print(model_list)
for model in model_list:
    metric_dict = {}
    print(f"Working on {model}.", flush=True)
    sm = Slisemap.load(Path(model_folder) / model, map_location=device)
    for n, f in metrics.items():
        val_sm = f(sm)
        metric_dict[n] = val_sm
        print(f"\t{model} {n}: {val_sm}")
    metric_dict["model"] = model
    dicts.append(metric_dict)
results = pd.DataFrame.from_dict(dicts)
today = datetime.today().date()
dataset_name = model_list[0].split("_")[0]
result_dir = curr_path.parent / f"results/{dataset_name}/{today}/"
result_dir.mkdir(parents=True, exist_ok=True)
results.to_pickle(result_dir / f"DR_comp_{random_seed}.pkl.gz")
