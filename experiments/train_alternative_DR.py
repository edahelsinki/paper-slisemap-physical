"""This script is used to train Slisemap and alternative DR methods for comparison.
It is mean to be run using slurm."""
import numpy as np
import torch
import sys
from pathlib import Path

curr_path = Path(__file__)
import time
from datetime import datetime
from slisemap import Slisemap
from slisemap.local_models import LogisticRegression
from slisemap.utils import LBFGS
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
import gc

sys.path.append(str(curr_path.parent.parent.parent / "notes"))

debug = True
# process input args
dataset_name = sys.argv[2]
job_id = int(sys.argv[1]) - 1
assert dataset_name in [
    "qm9",
    "gecko",
    "jets",
], f"Dataset {dataset_name} not implemented!"

# set up device
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    print("GPU not available, reverting to CPU!", flush=True)
    device = torch.device("cpu")

# set shared parameters
random_seed = range(1618033, 1618043)[job_id]
rng = np.random.default_rng(random_seed)
sample_size = 10000

# create folders
today = datetime.today().date()
# today = '2023-10-16'
models = curr_path.parent / f"models/{dataset_name}/{today}/"
models.mkdir(parents=True, exist_ok=True)

# set up slisemap hyperparameters
sm_hyperparams = {
    "device": device,
    "random_state": random_seed,
}
if dataset_name == "qm9":
    sm_hyperparams["lasso"] = 0.01
    sm_hyperparams["ridge"] = 0.001
elif dataset_name == "jets":
    sm_hyperparams["lasso"] = 0.0001
    sm_hyperparams["ridge"] = 0.0001
    sm_hyperparams["local_model"] = LogisticRegression
elif dataset_name == "gecko":
    sm_hyperparams["lasso"] = 0.001
    sm_hyperparams["ridge"] = 0.001


def load_sm(dataset_name):
    """Load trained slisemap model based on dataset name."""
    if dataset_name == "qm9":
        model_path = (
            curr_path.parent.parent.parent / "SI/models/qm9_nn_10k_0.01_0.001.sm"
        )
    elif dataset_name == "jets":
        model_path = (
            curr_path.parent.parent.parent / "SI/models/jets_rf_10k_0.0001_0.0001.sm"
        )
    elif dataset_name == "gecko":
        model_path = (
            curr_path.parent.parent.parent / "SI/GeckoQ/SM_32k_ExpF_real_zero_mean.sm"
        )
    sm = Slisemap.load(model_path, device=device)
    return sm


def load_data_from_sm(sm: Slisemap):
    """Get data from a Slisemap model."""
    X, Y = sm.get_X(intercept=False, numpy=False), sm.get_Y(numpy=False)
    return X, Y


def get_data(dataset_name, n_samples):
    """Scramble data."""
    # load dataset
    sm0 = load_sm(dataset_name)
    X, Y = load_data_from_sm(sm0)
    # ensure data is in GPU (if available)
    if torch.cuda.is_available():
        Y = Y.to(device)
        X = X.to(device)
    # scramble data
    random_indices = rng.permutation(len(X))
    X = X[random_indices, :]
    Y = Y[random_indices]
    return X[:n_samples, :], Y[:n_samples]


def optim_with_embedding(embedding):
    """Return a function to train Slisemap based on alternative embedding."""

    def train_fn(sm: Slisemap):
        sm.z_norm = 0
        X = sm.get_X()
        X = X[:, np.std(X, 0) != 0]
        Z = embedding
        sm._Z = torch.as_tensor(Z, **sm.tensorargs)
        sm._normalise()
        B = sm.get_B(numpy=False).requires_grad_(True)
        loss = sm.get_loss_fn()
        LBFGS(lambda: loss(sm.X, sm.Y, B, sm.Z), [B])
        sm._B = B.detach()

    return train_fn


def train_DR(model, Xs, **hyperparams):
    """Wrapper for training a DR method."""
    m = model(**hyperparams)
    embedding = m.fit_transform(Xs)
    return embedding


model_list = [
    (PCA, {"random_state": random_seed, "n_components": 2}),
    (TSNE, {"random_state": random_seed, "perplexity": 10}),
    (TSNE, {"random_state": random_seed, "perplexity": 20}),
    (TSNE, {"random_state": random_seed, "perplexity": 30}),
    (TSNE, {"random_state": random_seed, "perplexity": 40}),
    (TSNE, {"random_state": random_seed, "perplexity": 50}),
    (UMAP, {"random_state": random_seed, "n_neighbors": 5}),
    (UMAP, {"random_state": random_seed, "n_neighbors": 10}),
    (UMAP, {"random_state": random_seed, "n_neighbors": 15}),
    (UMAP, {"random_state": random_seed, "n_neighbors": 20}),
    (UMAP, {"random_state": random_seed, "n_neighbors": 25}),
]

# main training loop
gc.collect()
torch.cuda.empty_cache()
if debug:
    print(f"\tAllocated memory: {torch.cuda.memory_allocated(0)/1e6:.1f} GB")
    print(
        f"\tFree memory: {(torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)) / 1e6:.1f} MB"
    )
total_start = time.time()
print(f"{dataset_name}: Begin training ({job_id}, seed {random_seed}).", flush=True)
# get the data
Xs, ys = get_data(dataset_name, 10000)
gc.collect()
torch.cuda.empty_cache()
if debug:
    print(f"\tAllocated memory: {torch.cuda.memory_allocated(0)/1e6:.1f} GB")
    print(
        f"\tFree memory: {(torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)) / 1e6:.1f} MB"
    )
# train regular slisemap.
fname_slisemap = models / Path(f"{dataset_name}_slisemap_{random_seed}.sm")
if not fname_slisemap.exists():
    print("\tTrain regular slisemap.", flush=True)
    start_time = time.time()
    sm = Slisemap(Xs, ys, **sm_hyperparams)
    sm.optimise()
    print(
        f"\tRegular slisemap training done, duration {time.time() - start_time:.1f} s.",
        flush=True,
    )
    sm.save(fname_slisemap)
else:
    print("\tRegular slisemap already found, not recomputing.")
# train the alternative DR methods
for model, params in model_list:
    hyper_str = "_".join([f"{k}_{v}" for k, v in params.items()])
    fname_model_em = models / Path(f"{dataset_name}_{model.__name__}_{hyper_str}.npy")
    embedding = None
    if not fname_model_em.exists():
        start = time.time()
        print(f"\tTrain {model.__name__} ({hyper_str}).", flush=True)
        embedding = train_DR(model, Xs.cpu(), **params)
        with open(fname_model_em, "wb+") as f:
            np.save(f, embedding)
        print(f"\t\tEmbedding done, took {time.time() - start:.1f}s.")
    else:
        print(f"\tFound {model.__name__} ({hyper_str}), not retraining.", flush=True)
    fname_model_sm = models / Path(f"{dataset_name}_{model.__name__}_{hyper_str}.sm")
    if not fname_model_sm.exists():
        if embedding is None:
            with open(fname_model_em, "rb") as f:
                embedding = np.load(f)
        start = time.time()
        print(f"\tTrain slisemap on {model.__name__} ({hyper_str}).", flush=True)
        sm_c = Slisemap.load(fname_slisemap)
        optim_with_embedding(embedding)(sm_c)
        sm_c.save(fname_model_sm)
        print(f"\t\tSM on embedding done, took {time.time() - start:.1f}s.")
    else:
        print(
            f"\tFound slisemap for {model.__name__} ({hyper_str}), not retraining.",
            flush=True,
        )
print(f"In total, took {time.time() - total_start:.1f} s.")
