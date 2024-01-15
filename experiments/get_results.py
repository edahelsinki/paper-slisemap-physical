from slisemap import Slisemap
import numpy as np
import torch
import pickle
import sys

# sys.path.append('/home/seplauri/Documents/edahelsinki/molecular2022/notes')
from pathlib import Path

curr_path = Path(__file__)
sys.path.append(str(curr_path.parent.parent.parent / "notes"))
from comp_tools import *
import matplotlib.pyplot as plt
import time
import gc
from datetime import datetime
from collections import defaultdict
import os

models = sys.argv[2]  # model directory
results = sys.argv[3]  # results directory
results = Path(results)
results.mkdir(parents=True, exist_ok=True)
test_name = sys.argv[4]
comparison_style = sys.argv[5]  # which comparison style should be used
compute = True
n_runs = 10
job_id = int(sys.argv[1]) - 1
array_size = 10
save_chunks = True

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    print("GPU not available, reverting to CPU!")
    device = torch.device("cpu")


def load_models(model_dir: Path, comparison_style: str):
    norm_files = sorted([x for x in os.listdir(model_dir) if "_normal" in x])
    perm_files = sorted([x for x in os.listdir(model_dir) if "_perm" in x])
    half_files = sorted([x for x in os.listdir(model_dir) if "_half" in x])
    if comparison_style == "permutation":
        normal_models = defaultdict(list)
        perm_models = defaultdict(list)
        for f in norm_files:
            size = int(f.split("_")[1])
            normal_models[size].append(
                Slisemap.load(f"{model_dir}/{f}", map_location=device)
            )
        for f in perm_files:
            size = int(f.split("_")[1])
            perm_models[size].append(
                Slisemap.load(f"{model_dir}/{f}", map_location=device)
            )
        normal_models = {k: v for k, v in normal_models.items()}
        sample_sizes = sorted(normal_models.keys())
        perm_models = {k: v for k, v in perm_models.items()}
        cohorts = {"normal": normal_models, "permuted": perm_models}
    elif comparison_style == "versus":
        cohorts = defaultdict(list)
        for f in norm_files:
            size = int(f.split("_")[1])
            cohorts[size].append(
                Slisemap.load(f"{model_dir}/{f}", map_location=torch.device("cpu"))
            )
        sample_sizes = sorted(cohorts.keys())
        cohorts = {k: v for k, v in cohorts.items()}
    elif comparison_style == "half":
        # need two cohorts here
        normal_models = defaultdict(list)
        half_models = defaultdict(list)
        for f in norm_files:
            size = int(f.split("_")[1])
            normal_models[size].append(
                Slisemap.load(f"{model_dir}/{f}", map_location=device)
            )
        for f in half_files:
            size = int(f.split("_")[1])
            half_models[size].append(
                Slisemap.load(f"{model_dir}/{f}", map_location=device)
            )
        normal_models = {k: v for k, v in normal_models.items()}
        half_models = {k: v for k, v in half_models.items()}
        sample_sizes = sorted(normal_models.keys())
        cohorts = {"normal": normal_models, "half": half_models}
    elif comparison_style == "half_perm":
        # need two cohorts here
        perm_models = defaultdict(list)
        half_models = defaultdict(list)
        for f in perm_files:
            size = int(f.split("_")[1])
            perm_models[size].append(
                Slisemap.load(f"{model_dir}/{f}", map_location=device)
            )
        for f in half_files:
            size = int(f.split("_")[1])
            half_models[size].append(
                Slisemap.load(f"{model_dir}/{f}", map_location=device)
            )
        perm_models = {k: v for k, v in perm_models.items()}
        half_models = {k: v for k, v in half_models.items()}
        sample_sizes = sorted(perm_models.keys())
        cohorts = {"permuted": perm_models, "half": half_models}

    return cohorts, sample_sizes


def run_test(test_function, comparison_style, cohorts, sample_size, **test_kwargs):
    if comparison_style == "versus":
        sms = cohorts[sample_size]
        res = torch.zeros((n_runs, n_runs))
        for i in range(n_runs):
            for j in range(i, n_runs):
                check = (
                    (n_runs * (n_runs - 1) / 2)
                    - (n_runs - i) * ((n_runs - i) - 1) / 2
                    + j
                    - i
                    - 1
                )
                if check % array_size != job_id:
                    continue
                res[i, j] = test_function(sms[i], sms[j], **test_kwargs)
    elif comparison_style == "half":
        sms_normal = cohorts["normal"][sample_size]
        sms_half = cohorts["half"][sample_size]
        res = torch.zeros(n_runs)
        for i in range(n_runs):
            if i % array_size != job_id:
                continue
            res[i] = test_function(sms_normal[i], sms_half[i], **test_kwargs)
    elif comparison_style == "permutation":
        sms_normal = cohorts["normal"][sample_size]
        sms_perm = cohorts["permuted"][sample_size]
        res = torch.zeros(n_runs)
        for i in range(n_runs):
            if i % array_size != job_id:
                continue
            res[i] = test_function(sms_normal[i], sms_perm[i], **test_kwargs)
    elif comparison_style == "half_perm":
        sms_half = cohorts["half"][sample_size]
        sms_perm = cohorts["permuted"][sample_size]
        res = torch.zeros(n_runs)
        for i in range(n_runs):
            if i % array_size != job_id:
                continue
            res[i] = test_function(sms_half[i], sms_perm[i], **test_kwargs)
    return res


def set_double_distance_wrapper(sm1: Slisemap, sm2: Slisemap):
    L1 = sm1.get_L(numpy=False)
    L2 = sm2.get_L(numpy=False)
    D1 = sm1.get_D(numpy=False)
    D2 = sm2.get_D(numpy=False)
    Dnew1 = set_double_distance(L1, L2, D1)
    Dnew2 = set_double_distance(L1, L2, D2)
    out = (Dnew1.mean() + Dnew2.mean()).item() / 2.0
    return out


def set_half_distance_wrapper(sm1: Slisemap, sm2: Slisemap):
    L1 = sm1.get_L(numpy=False)
    L2 = sm2.get_L(numpy=False)
    D1 = sm1.get_D(numpy=False)
    D2 = sm2.get_D(numpy=False)
    Dnew1 = set_half_distance(L1, D2)
    Dnew2 = set_half_distance(L2, D1)
    out = (Dnew1.mean() + Dnew2.mean()).item() / 2.0
    return out


def training_quality(sm_normal: Slisemap, sm_perm: Slisemap):
    # Y = sm_normal.get_Y(numpy=False)
    # Yhat = sm_normal.predict(numpy=False)
    # L = sm_normal.local_loss(Yhat, Y)
    # L0 = sm_normal.local_loss(sm_perm.predict(numpy=False), Y)
    # out = (L / L0).median().item()
    L = sm_normal.value(individual=False, numpy=False)
    L0 = sm_perm.value(individual=False, numpy=False)
    out = (L / L0).item()
    return out


print("Load models.", flush=True)
cohorts, sample_sizes = load_models(models, comparison_style=comparison_style)
for k in sample_sizes:
    start = time.time()
    if k > 17000:
        continue
    print(f"Calculate {test_name} for {k} samples.", flush=True)
    match test_name:
        case "local_model_distance":
            # using full models
            fname = (
                results / f"local_model_distance_{comparison_style}_{k}_{job_id}.pkl"
            )
            if fname.exists():
                print("\tFound existing result, not recomputing!", flush=True)
                continue
            B_d_m = run_test(
                B_distance,
                comparison_style=comparison_style,
                cohorts=cohorts,
                sample_size=k,
                include_y=True,
                match_by_model=True,
            )
            print(f"Mean: {B_d_m[B_d_m != 0.].mean().item()}")
            with open(fname, "wb") as f:
                torch.save(B_d_m, f, pickle_protocol=pickle.HIGHEST_PROTOCOL)
            print(f"\tSaved to {fname}")
        case "set_half_distance":
            fname = results / f"set_half_distance_{comparison_style}_{k}_{job_id}.pkl"
            if fname.exists():
                continue
            s_h_d = run_test(
                set_half_distance_wrapper,
                comparison_style=comparison_style,
                cohorts=cohorts,
                sample_size=k,
            )

            with open(fname, "wb") as f:
                torch.save(s_h_d, f, pickle_protocol=pickle.HIGHEST_PROTOCOL)
            print(f"\tSaved to {fname}")
        case "set_double_distance":
            fname = results / f"set_double_distance_{comparison_style}_{k}_{job_id}.pkl"
            if fname.exists():
                print("\tFound existing result, not recomputing!", flush=True)
                continue
            s_d_d = run_test(
                set_double_distance_wrapper,
                comparison_style=comparison_style,
                cohorts=cohorts,
                sample_size=k,
            )

            with open(fname, "wb") as f:
                torch.save(s_d_d, f, pickle_protocol=pickle.HIGHEST_PROTOCOL)
            print(f"\tSaved to {fname}")
        case "neighbourhood_distance":
            fname = (
                results / f"neighbourhood_distance_{comparison_style}_{k}_{job_id}.pkl"
            )
            if fname.exists():
                print("\tFound existing result, not recomputing!", flush=True)
                continue
            E_d = run_test(
                faster_epsilon_ball,
                comparison_style=comparison_style,
                cohorts=cohorts,
                sample_size=k,
                debug=True
            )
            print(f"Mean: {E_d[E_d != 0.].mean().item()}")
            with open(fname, "wb") as f:
                torch.save(E_d, f, pickle_protocol=pickle.HIGHEST_PROTOCOL)
            print(f"\tSaved to {fname}")
            # calculate neighbourhood distance
            # using half-shared points
        case "training_quality":
            fname = results / f"training_quality_{comparison_style}_{k}_{job_id}.pkl"
            if fname.exists():
                print("\tFound existing result, not recomputing!", flush=True)
                continue
            Q_t = run_test(
                training_quality,
                comparison_style=comparison_style,
                cohorts=cohorts,
                sample_size=k,
            )
            with open(fname, "wb") as f:
                torch.save(Q_t, f, pickle_protocol=pickle.HIGHEST_PROTOCOL)
            print(f"\tSaved to {fname}")
            # calculate trainging quality
            # using perm and normal models
        case _:
            raise ValueError(f"{test_name} not implemented!")
    print(f"Calculating took {time.time() - start:.3f} s.")
# with open(f'{results}/B_ds_{job_id}.pkl', 'wb') as f:
#     pickle.dump(B_ds, f, protocol=pickle.HIGHEST_PROTOCOL)
# with open(f'{results}/C_Bs_{job_id}.pkl', 'wb') as f:
#     pickle.dump(C_Bs, f, protocol=pickle.HIGHEST_PROTOCOL)
# with open(f'{results}/C_Js_{job_id}.pkl', 'wb') as f:
#     pickle.dump(C_Js, f, protocol=pickle.HIGHEST_PROTOCOL)
# with open(f'{results}/E_ds_{job_id}.pkl', 'wb') as f:
#     pickle.dump(E_ds, f, protocol=pickle.HIGHEST_PROTOCOL)
