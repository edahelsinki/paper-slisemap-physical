from slisemap import Slisemap
import pandas as pd
import numpy as np
import torch
from pathlib import Path

work_directory = Path(__file__).parent.parent / "GeckoQ/"
model_directory = Path(__file__).parent

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    print("GPU not available, reverting to CPU!", flush=True)
    device = torch.device("cpu")

hyperparams = {"device": device, "lasso": 0.001, "ridge": 0.001}


def load_geckoQ():
    data = pd.read_pickle(work_directory / "Gecko_Q_expF_published_zero_mean.pkl.gz")
    variables = [x for x in data.columns if x != "pSat_Pa"]
    X = torch.tensor(data[variables].to_numpy(), dtype=torch.float32)
    Y = torch.atleast_2d(
        torch.tensor(data["pSat_Pa"].to_numpy(), dtype=torch.float32)
    ).T
    return X, Y, variables


if __name__ == "__main__":
    print("Load data.")
    X, Y, variables = load_geckoQ()
    print("Init Slisemap.")
    sm = Slisemap(X, Y, intercept=True, **hyperparams)
    sm.metadata.set_variables(variables)
    print("Optimise.")
    sm.optimise(verbose=True)
    print("Save trained model to disk.")
    sm.save(work_directory / "geckoq_32k_expf_0.001_0.001.sm")
    print("Done.")
