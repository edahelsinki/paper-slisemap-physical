from slisemap import Slisemap
import numpy as np
import torch
import sys
from pathlib import Path

curr_path = Path(__file__)
import time
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import os
from slisemap.local_models import LogisticRegression

sys.path.append(str(curr_path.parent.parent.parent))
from data import load_qm9, load_jets


sys.path.append(str(curr_path.parent.parent.parent / "notes"))

# process input args
job_id = int(sys.argv[1]) - 1
dataset_name = sys.argv[2]
assert dataset_name in [
    "qm9",
    "gecko",
    "jets",
], f"Dataset {dataset_name} not implemented!"

# set shared parameters
random_seed = 1618033
rng = np.random.default_rng(random_seed)
sample_sizes = np.round(np.logspace(np.log10(100), np.log10(16000), 10)).astype(int)

# set up device
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    print("GPU not available, reverting to CPU!", flush=True)
    device = torch.device("cpu")

# set hyperparameters
hyperparams = {
    "device": device,
    "random_state": random_seed,
}
if dataset_name == "qm9":
    hyperparams["lasso"] = 0.01
    hyperparams["ridge"] = 0.001
elif dataset_name == "jets":
    hyperparams["lasso"] = 0.0001
    hyperparams["ridge"] = 0.0001
    hyperparams["local_model"] = LogisticRegression
elif dataset_name == "gecko":
    hyperparams["lasso"] = 0.001
    hyperparams["ridge"] = 0.001


# create folders
today = datetime.today().date()
models = curr_path.parent / f"models/{dataset_name}/{today}/"
models.mkdir(parents=True, exist_ok=True)


def load_data(dataset_name):
    if dataset_name == "qm9":
        X, Y = load_qm9(model="nn")
        scale_X = StandardScaler()
        X = scale_X.fit_transform(X)
        scale_Y = StandardScaler()
        Y = scale_Y.fit_transform(Y)
        X, Y = torch.tensor(X, dtype=torch.float32), torch.tensor(
            Y, dtype=torch.float32
        )
        # remove nunique values
        Z = torch.hstack([X, Y])
        Z = torch.unique(Z, dim=0)
        X = Z[:, :-1]
        Y = Z[:, -1:]
    elif dataset_name == "jets":
        X, Y = load_jets(model="rf")
        scale_X = StandardScaler()
        X = scale_X.fit_transform(X)
        X, Y = torch.tensor(X, dtype=torch.float32), torch.tensor(
            Y, dtype=torch.float32
        )
    elif dataset_name == "gecko":
        sm_full = Slisemap.load(
            curr_path.parent.parent / "models/28k/SM_28k_ExpF_real_1e-3_1e-3_V.sm",
            device=device,
        )
        X = sm_full.get_X(intercept=False, numpy=False)
        Y = sm_full.get_Y(numpy=False)
    return X, Y


def get_data(dataset_name, n_samples):
    # load dataset
    X, Y = load_data(dataset_name)
    # ensure data is in GPU (if available)
    if torch.cuda.is_available():
        Y = Y.to(device)
        X = X.to(device)
    # scramble data for subsampling
    random_indices = rng.permutation(len(X))
    X = X[random_indices, :]
    Y = Y[random_indices]
    # get subset of data
    start_idx = (job_id * max(sample_sizes)) % len(X)
    if (start_idx + n_samples) < len(X):
        X_s = X[start_idx : start_idx + n_samples, :]
        Y_s = Y[start_idx : start_idx + n_samples]
    else:
        X_s = torch.vstack(
            (X[start_idx:, :], X[0 : n_samples - (len(X) - start_idx), :])
        )
        Y_s = torch.vstack((Y[start_idx:], Y[0 : n_samples - (len(X) - start_idx)]))
    # get another subset of data, half overlapping with the previous
    start_idx = (job_id * max(sample_sizes) + (n_samples // 2)) % len(X)
    if (start_idx + n_samples) < len(X):
        X_h = X[start_idx : start_idx + n_samples, :]
        Y_h = Y[start_idx : start_idx + n_samples]
    else:
        X_h = torch.vstack(
            (X[start_idx:, :], X[0 : n_samples - (len(X) - start_idx), :])
        )
        Y_h = torch.vstack((Y[start_idx:], Y[0 : n_samples - (len(X) - start_idx)]))
    return X_s, Y_s, X_h, Y_h


# main training loop
for n_samples in sample_sizes:
    print(
        f"Job ID {job_id+1} starting on {dataset_name} with "
        + f"n_samples={n_samples}",
        flush=True,
    )
    start = time.time()
    print("\tStarting normal training.", flush=True)
    Xs, ys, Xh, yh = get_data(dataset_name, n_samples)
    fname_normal = models / Path(
        f'{dataset_name}_{n_samples}_{job_id}_normal_lasso_{hyperparams["lasso"]}_ridge_{hyperparams["ridge"]}.sm'
    )
    if not fname_normal.exists():
        sm_normal = Slisemap(Xs, ys, **hyperparams)
        sm_normal.optimise()
        sm_normal.save(fname_normal)
        print(
            f"\tNormal training done, duration {time.time() - start:.3f} s.", flush=True
        )
    else:
        print("\t\tFound existing model, not recomputing!", flush=True)

    fname_half = models / Path(
        f'{dataset_name}_{n_samples}_{job_id}_half_lasso_{hyperparams["lasso"]}_ridge_{hyperparams["ridge"]}.sm'
    )
    start_half = time.time()
    print("\tStarting .5 shared points.", flush=True)
    if not fname_half.exists():
        sm_half = Slisemap(Xh, yh, **hyperparams)
        sm_half.optimise()

        sm_half.save(fname_half)
        print(
            f"\t.5 training done, duration {time.time() - start_half:.3f} s.",
            flush=True,
        )
    else:
        print("\t\tFound existing model, not recomputing!", flush=True)
        continue
    fname_perm = models / Path(
        f'{dataset_name}_{n_samples}_{job_id}_perm_lasso_{hyperparams["lasso"]}_ridge_{hyperparams["ridge"]}.sm'
    )
    start_perm = time.time()
    print("\tStarting random permutation training.", flush=True)
    if not fname_perm.exists():
        yp = ys[rng.permutation(len(ys))]
        sm_perm = Slisemap(Xs, yp, **hyperparams)
        sm_perm.optimise()

        sm_perm.save(fname_perm)
        print(
            f"\tPermutation training done, duration {time.time() - start_perm:.3f} s.",
            flush=True,
        )
    else:
        print("\t\tFound existing model, not recomputing!", flush=True)
    print(
        f"Total {dataset_name} {n_samples} samples run {job_id+1} duration: {time.time() - start:.3f} s.",
        flush=True,
    )
