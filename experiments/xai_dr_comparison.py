"""XAI/DR method comparison. Meant to be run in a SLURM environment!"""
import numpy as np
import torch
import sys
from pathlib import Path

curr_path = Path(__file__)
import time
from slisemap import Slisemap
from slisemap.utils import LBFGS
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
import gc
import shap
import shap.maskers
from lime.lime_tabular import LimeTabularExplainer
from slisemap.local_models import LogisticRegression, LinearRegression
from slisemap.utils import global_model as _global_model
from slisemap.utils import tonp
from functools import partial
import slise
import warnings
from scipy.spatial.distance import cdist
import pandas as pd

sys.path.append(str(curr_path.parent.parent.parent))
from data import get_jets_bb, get_qm9_bb

sys.path.append(str(curr_path.parent.parent.parent / "notes"))


def global_model(sm: Slisemap) -> torch.Tensor:
    """Train global model from SLISEMAP."""
    return _global_model(
        X=sm.X,
        Y=sm.Y,
        local_model=sm.local_model,
        local_loss=sm.local_loss,
        coefficients=sm.coefficients,
        lasso=sm.lasso,
        ridge=sm.ridge,
    )


def get_hyperparameters(dataset_name, random_seed):
    """Set hyperparameters for SLISEMAP according to dataset name."""
    hyperparameters = {"device": device, "random_state": random_seed}
    if dataset_name == "qm9":
        hyperparameters["lasso"] = 0.01
        hyperparameters["ridge"] = 0.001
    elif dataset_name == "jets":
        hyperparameters["lasso"] = 0.0001
        hyperparameters["ridge"] = 0.0001
        hyperparameters["local_model"] = LogisticRegression
    elif dataset_name == "gecko":
        hyperparameters["lasso"] = 0.001
        hyperparameters["ridge"] = 0.001
    return hyperparameters


def load_sm(dataset_name):
    """Load SLISEMAP object from file."""
    if dataset_name == "qm9":
        model_path = (
            curr_path.parent.parent.parent / "SI/models/qm9_nn_10k_x_35_0.01_0.001.sm"
        )
    elif dataset_name == "jets":
        model_path = (
            curr_path.parent.parent.parent / "SI/models/jets_rf_10k_0.0001_0.0001.sm"
        )
    elif dataset_name == "gecko":
        model_path = (
            curr_path.parent.parent.parent / "SI/models/geckoq_32k_expf_0.001_0.001.sm"
        )
    sm = Slisemap.load(model_path, device="cpu")
    return sm


def load_data_from_sm(sm: Slisemap):
    """Extract data from SLISEMAP."""
    X, Y = sm.get_X(intercept=False, numpy=False), sm.get_Y(numpy=False)
    return X, Y


def get_data(dataset_name, n_samples, rng, scaler=False):
    """Load and preprocess data from SLISEMAP."""
    # load dataset
    sm0 = load_sm(dataset_name)
    X, Y = load_data_from_sm(sm0)
    # scramble data
    random_indices = rng.permutation(len(X))
    X = X[random_indices, :]
    Y = Y[random_indices]
    if scaler:
        return (
            X[:n_samples, :],
            Y[:n_samples],
            sm0.metadata["X_center"][None, :],
            sm0.metadata["X_scale"][None, :],
            sm0.metadata.get("Y_center"),
            sm0.metadata.get("Y_scale"),
        )
    else:
        return X[:n_samples, :], Y[:n_samples]


def optim_with_embedding(embedding):
    """Optimize SLISEMAP local models based on an embedding."""

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


def prepare_DR(model, X, y, **hyperparams):
    """Train a DR method."""
    X = tonp(X)
    y = tonp(y)
    t1 = time.perf_counter(), time.process_time()
    m = model(**hyperparams)
    embedding = m.fit_transform(X)
    # fit slisemap object
    sm = Slisemap(X, y, **get_hyperparameters(dataset_name, random_seed=m.random_state))
    optim_with_embedding(embedding)(sm)
    t2 = time.perf_counter(), time.process_time()
    prep_wall, prep_cpu = t2[0] - t1[0], t2[1] - t1[1]
    return slisemap_predict(sm), sm, prep_wall, prep_cpu


def get_slise(X, y, classifier, epsilon: float, hyperparameters):
    X = tonp(X)
    y = tonp(y)
    if classifier:
        y = y[:, 0]
    t1 = time.perf_counter(), time.process_time()
    s = slise.SliseExplainer(
        X,
        y.ravel(),
        epsilon**0.5,
        logit=classifier,
        lambda1=hyperparameters["lasso"],
        lambda2=hyperparameters["ridge"],
        # debug=True,
        num_threads=1,
    )
    t2 = time.perf_counter(), time.process_time()

    def explain(i):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s.explain(i)
        if classifier:
            return lambda X: np.stack(
                (s.predict(tonp(X)), 1.0 - s.predict(tonp(X))), -1
            )
        return lambda X: s.predict(tonp(X))

    return explain, t2[0] - t1[0], t2[1] - t1[1]


def get_slisemap(X, y, hyperparameters):
    t1 = time.perf_counter(), time.process_time()
    sm = Slisemap(X, y, **hyperparameters)
    sm.optimise()
    t2 = time.perf_counter(), time.process_time()
    B = sm.get_B(False)
    return (
        slisemap_predict(sm),
        sm,
        t2[0] - t1[0],
        t2[1] - t1[1],
    )


def get_lime(X, y, pred_fn, disc=True, classifier=False):
    X = tonp(X)
    y = tonp(y)
    t1 = time.perf_counter(), time.process_time()
    explainer = LimeTabularExplainer(
        X,
        "classification" if classifier else "regression",
        y,
        discretize_continuous=disc,
    )
    t2 = time.perf_counter(), time.process_time()

    def explain(i):
        exp = explainer.explain_instance(X[i, :], pred_fn, num_samples=5_000)
        b = np.zeros((1, X.shape[1]))
        for j, v in exp.as_map()[1]:
            b[0, j] = v
        inter = exp.intercept[1]
        if disc:
            di = explainer.discretizer.discretize(X)[i, :]

        def predict(X):
            X = tonp(X)
            if disc:
                # Lime works on discretised data, so we need to discretise the data
                # to be able to apply the linear model, and use lime for prediction.
                X = di == explainer.discretizer.discretize(X)
            Y = np.sum(X * b, -1, keepdims=True) + inter
            if classifier:
                Y = np.clip(Y, 0.0, 1.0)
                Y = np.concatenate((1.0 - Y, Y), -1)
            return Y

        return predict

    return explain, t2[0] - t1[0], t2[1] - t1[1]


def get_shap(X, y, pred_fn, partition=True, classifier=False):
    X = tonp(X)
    y = tonp(y)
    if classifier:
        link = shap.links.logit
        y = y[:, 0]
        old_pred = pred_fn
        pred_fn = lambda X: old_pred(X)[:, 0]
    else:
        link = shap.links.identity
    if partition:
        t1 = time.perf_counter(), time.process_time()
        masker = shap.maskers.Partition(X, max_samples=1_000)
        explainer = shap.explainers.Partition(
            pred_fn, masker=masker, link=link, linearize_link=False
        )
        t2 = time.perf_counter(), time.process_time()

        def explain(i):
            shapX = X[None, i, :]
            exp = explainer(shapX, silent=True)
            b = exp.values.reshape((exp.values.shape[0], -1))
            inter = float(exp.base_values)
            scale = _shap_get_scale(X, y, shapX, b, inter, link)
            return lambda X: _shap_predict(X, shapX, scale, b, inter, classifier)

        return explain, t2[0] - t1[0], t2[1] - t1[1]
    else:
        t1 = time.perf_counter(), time.process_time()
        explainer = shap.explainers.Sampling(pred_fn, X)
        t2 = time.perf_counter(), time.process_time()

        def explain(i):
            shapX = X[None, i, :]
            b = explainer.shap_values(shapX, silent=True)
            inter = explainer.expected_value
            scale = _shap_get_scale(X, y, shapX, b, inter, link)
            return lambda X: _shap_predict(X, shapX, scale, b, inter, classifier)

        return explain, t2[0] - t1[0], t2[1] - t1[1]


def _shap_predict(X, shapX, scale, b, intercept, classifier):
    X = tonp(X)
    dist = (X - shapX) ** 2
    kernel = np.exp(-(np.abs(scale) + 1e-6) * dist)
    P = np.sum(kernel * b, -1, keepdims=True) + intercept
    if classifier:
        P = shap.links.logit.inverse(P)
        return np.squeeze(np.stack((P, 1.0 - P), -1))
    return P


def _shap_get_scale(X, y, shapX, b, intercept, link):
    dist = torch.as_tensor((X - shapX) ** 2, dtype=torch.float32)
    scale = torch.ones_like(dist[:1, :], requires_grad=True)
    Y = torch.as_tensor(link(y), dtype=dist.dtype)
    if len(Y.shape) == 1:
        Y = Y[:, None]
    b = torch.as_tensor(b, dtype=dist.dtype)

    def loss():
        kernel = torch.exp(-(torch.abs(scale) + 1e-6) * dist)
        P = torch.sum(kernel * b, -1, keepdim=True) + intercept
        return torch.mean(torch.abs(Y - P))

    LBFGS(loss, [scale])
    return tonp(scale)


def profile(function):
    """Profile a given function."""
    # empty caches
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # time optimisation
    t1 = time.perf_counter(), time.process_time()
    val = function()
    t2 = time.perf_counter(), time.process_time()
    if val is not None:
        return val, t2[0] - t1[0], t2[1] - t1[1]
    return t2[0] - t1[0], t2[1] - t1[1]


def slisemap_predict(sm):
    B = sm.get_B(numpy=False)
    return lambda i: lambda X: tonp(sm.local_model(sm._as_new_X(X), B[None, i])[0, ...])


def get_methods(dataset_name, random_seed, X, y, epsilon, cls, pred_fn=None):
    model_list = [
        (
            "PCA",
            partial(
                prepare_DR,
                model=PCA,
                X=X,
                y=y,
                random_state=random_seed,
                n_components=2,
            ),
        ),
        (
            "TSNE",
            partial(
                prepare_DR,
                model=TSNE,
                X=X,
                y=y,
                random_state=random_seed,
                perplexity=30,
            ),
        ),
        (
            "UMAP",
            partial(
                prepare_DR,
                model=UMAP,
                X=X,
                y=y,
                random_state=random_seed,
                n_neighbors=15,
            ),
        ),
        (
            "SLISEMAP",
            partial(
                get_slisemap,
                X=X.to(device=device),
                y=y.to(device=device),
                hyperparameters=get_hyperparameters(dataset_name, random_seed),
            ),
        ),
    ]
    if pred_fn != None:
        model_list += [
            ("SHAP", partial(get_shap, X=X, y=y, pred_fn=pred_fn, classifier=cls)),
            ("LIME", partial(get_lime, X=X, y=y, pred_fn=pred_fn, classifier=cls)),
            (
                "LIME (nd)",
                partial(
                    get_lime, X=X, y=y, pred_fn=pred_fn, disc=False, classifier=cls
                ),
            ),
        ]
    model_list += [
        (
            "SLISE",
            partial(
                get_slise,
                X=X,
                y=y,
                classifier=cls,
                epsilon=epsilon,
                hyperparameters=get_hyperparameters(dataset_name, random_seed),
            ),
        ),
    ]
    return model_list


def evaluate(job_id, dataset_name, sample_size, result_dir, debug=False):
    # main training loop

    if debug:
        print(f"\tAllocated memory: {torch.cuda.memory_allocated(0)/1e6:.1f} GB")
        print(
            f"\tFree memory: {(torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)) / 1e6:.1f} MB"
        )
    total_start = time.time()
    random_seed = range(1618033, 1618043)[job_id]
    print(
        f"{dataset_name}: Begin training on {dataset_name} ({job_id}, seed {random_seed}).",
        flush=True,
    )
    file = result_dir / f"{dataset_name}_comparison_{job_id}.pkl.gz"
    file.parent.mkdir(parents=True, exist_ok=True)
    if file.exists():
        results = pd.read_pickle(file)
    else:
        results = pd.DataFrame()
    rng = np.random.default_rng(random_seed)
    if dataset_name in ["qm9", "jets"]:
        X, y, X_center, X_scale, y_center, y_scale = get_data(
            dataset_name, sample_size, rng, scaler=True
        )
    else:
        X, y = get_data(dataset_name, sample_size, rng)
    n_expl = 100
    cls = dataset_name == "jets"
    local_loss = LogisticRegression.loss if cls else LinearRegression.loss
    local_model = LogisticRegression.predict if cls else LinearRegression.predict
    sm_hyperparameters = get_hyperparameters(dataset_name, random_seed=random_seed)
    pred_fn = None
    if dataset_name == "jets":
        bb_fn = get_jets_bb("rf")
        pred_fn = lambda X: bb_fn((X * X_scale) + X_center)
        y = torch.tensor(pred_fn(X), device=X.device, dtype=X.dtype)
    elif dataset_name == "qm9":
        bb_fn = get_qm9_bb("nn")
        pred_fn = lambda X: (bb_fn((X * X_scale) + X_center) - y_center) / y_scale
        y = torch.tensor(pred_fn(X), device=X.device, dtype=X.dtype)
    # overwrite y with the BB values if BB is available
    global_model = _global_model(
        X=X,
        Y=y,
        local_model=local_model,
        local_loss=local_loss,
        coefficients=X.shape[1],
        lasso=sm_hyperparameters["lasso"],
        ridge=sm_hyperparameters["ridge"],
    )
    global_loss = local_loss(local_model(X, global_model), y)
    epsilon = torch.quantile(global_loss, 0.3).detach().item()
    for model_name, prepare_fun in get_methods(
        dataset_name, random_seed, X, y, epsilon, cls, pred_fn
    ):
        if ("method" in results) and (model_name in results["method"].values):
            print(f"Found {model_name} in {file}. Skipping.")
            continue
        print(f"Start {model_name}.", flush=True)
        # need to harmonize getting the times
        print("\tPreparing.", flush=True)
        if model_name in ["LIME", "LIME (nd)", "SHAP", "SLISE"]:
            explain, prep_wall, prep_cpu = prepare_fun()
            D = cdist(tonp(X[:n_expl, :]), tonp(X))
        else:
            explain, sm, prep_wall, prep_cpu = prepare_fun()
            D = sm.get_D(numpy=True)[:n_expl, :]
        D += np.eye(n_expl, X.shape[0]) * np.max(D)
        nn = np.argsort(D, 1)[:, : (sample_size // 10)]
        print(f"\tPreparation done. Took {prep_wall}s.", flush=True)
        tot_exp_wall = 0.0
        L = []
        print("\tExplaining.", flush=True)
        for i in range(n_expl):
            # profile predict function
            # add to loss matrix
            print(f"\tProduce explanation {i+1}/{n_expl}.", flush=True)
            pred, exp_wall, exp_cpu = profile(lambda: explain(i))
            tot_exp_wall += exp_wall
            yhat = torch.tensor(pred(X), device=X.device, dtype=X.dtype)
            if yhat.get_device() != -1:
                yhat = yhat.to(device="cpu")
            if yhat.dim() == 1:
                yhat = yhat[:, None]
            L.append(tonp(local_loss(yhat, y)))
        print(f"\tExplanations done. Took {tot_exp_wall}s.", flush=True)
        L = np.stack(L, 0)
        mean_exp_wall = tot_exp_wall / n_expl
        # calculate the metrics based on loss and distance
        res = dict(
            job=job_id,
            method=model_name,
            data=dataset_name,
            time_setup=prep_wall,
            time_explain=exp_wall,
            time_one=prep_wall / X.shape[0] + mean_exp_wall,
            time_all=prep_wall + tot_exp_wall,
            time_extrap=prep_wall + mean_exp_wall * sample_size,
            local_loss=L.diagonal().mean(),
            coverage=(L < epsilon).mean(),
            coverage_nn=(L[np.arange(L.shape[0])[:, None], nn] < epsilon).mean(),
            stability=L[np.arange(L.shape[0])[:, None], nn].mean(),
            epsilon=epsilon,
        )
        results = pd.concat((results, pd.DataFrame([res])), ignore_index=True)
        results.to_pickle(result_dir / f"{dataset_name}_comparison_{job_id}.pkl.gz")
    print(f"In total, took {time.time() - total_start:.1f} s.")


if __name__ == "__main__":
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
        # torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        print("GPU not available, reverting to CPU!", flush=True)
        device = torch.device("cpu")

    # set shared parameters
    sample_size = 5_000

    # create folders
    # today = datetime.today().date()
    result_dir = (
        curr_path.parent / f"results/{dataset_name}/xai_dr_comparison_fix_lime/"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    evaluate(job_id, dataset_name, sample_size, result_dir)
