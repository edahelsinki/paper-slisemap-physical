import matplotlib.pyplot as plt
from pathlib import Path
from os import listdir
import numpy as np
import torch
from collections import defaultdict
import io
import pickle
import matplotlib.ticker as mticker

curr_path = Path(__file__)
# input directories
res_dir = curr_path.parent / "results/"
jets_quality = res_dir / "jets/2023-06-09/"
qm9_quality = res_dir / "qm9/2023-06-09/"
gecko_quality = res_dir / "gecko/2023-06-09/"
jets_local_model_d = res_dir / "jets_smaller/2023-06-08/"
qm9_local_model_d = res_dir / "qm9_smaller/2023-06-08/"
gecko_local_model_d = res_dir / "gecko_smaller/2023-06-08/"
jets_neighbourhood_d = res_dir / "jets/2023-06-09/"
qm9_neighbourhood_d = res_dir / "qm9/2023-06-09/"
gecko_neighbourhood_d = res_dir / "gecko/2023-06-09/"
# output directory
plot_dir = curr_path.parent / "figures/"
plot_dir.mkdir(parents=True, exist_ok=True)


def load_results(result_dir, comparison_style, test_name):
    """Helper function to load datasets."""
    res_files = [
        x
        for x in listdir(result_dir)
        if (test_name in x) and (f"_{comparison_style}_" in x)
    ]
    if comparison_style == "half":
        res_files = [x for x in res_files if "_perm_" not in x]
    results = defaultdict(list)
    for file in res_files:
        sample_size = int(file.split("_")[-2])
        with open(result_dir / file, "rb") as f:
            results[sample_size].append(torch.load(f, map_location="cpu"))

    res = {}
    for k, v in results.items():
        res[k] = torch.sum(torch.stack(v), axis=-1)
    return res


def plot(dataset, ax, comparison_style, **plot_kwargs):
    """Helper function for plots."""
    sample_sizes = [x for x in sorted(dataset.keys()) if x < 17000]
    to_plot = []
    for s in sample_sizes:
        p = dataset[s]
        if comparison_style == "versus":
            p = p[p != 0].flatten()
        p = p[p != np.inf]
        to_plot.append(p.mean())
    ax.plot(sample_sizes, to_plot, **plot_kwargs)


# load results
ds_q_jets = load_results(jets_quality, "permutation", "training_quality")
ds_q_qm9 = load_results(qm9_quality, "permutation", "training_quality")
ds_q_gecko = load_results(gecko_quality, "permutation", "training_quality")
ds_B_dist_jets = load_results(
    jets_local_model_d,
    "versus",
    "local_model_distance",
)
ds_B_dist_qm9 = load_results(
    qm9_local_model_d,
    "versus",
    "local_model_distance",
)
ds_B_dist_gecko = load_results(
    gecko_local_model_d,
    "versus",
    "local_model_distance",
)
ds_B_dist_jets_p = load_results(
    jets_local_model_d,
    "permutation",
    "local_model_distance",
)
ds_B_dist_qm9_p = load_results(
    qm9_local_model_d,
    "permutation",
    "local_model_distance",
)
ds_B_dist_gecko_p = load_results(
    gecko_local_model_d,
    "permutation",
    "local_model_distance",
)
ds_e_dist_jets = load_results(
    jets_neighbourhood_d,
    "half",
    "neighbourhood_distance",
)
ds_e_dist_jets_p = load_results(
    jets_neighbourhood_d,
    "half_perm",
    "neighbourhood_distance",
)
ds_e_dist_qm9 = load_results(
    qm9_neighbourhood_d,
    "half",
    "neighbourhood_distance",
)
ds_e_dist_qm9_p = load_results(
    qm9_neighbourhood_d,
    "half_perm",
    "neighbourhood_distance",
)
ds_e_dist_gecko = load_results(
    gecko_local_model_d,
    "half",
    "neighbourhood_distance",
)
ds_e_dist_gecko_p = load_results(
    gecko_local_model_d,
    "half_perm",
    "neighbourhood_distance",
)
# plot
fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(15, 5))
xticks = [100, 500, 1000, 5000, 10000, 15000]
plot(ds_q_gecko, axs[0], "permutation", label="Gecko", marker="P", color="tab:blue")
plot(ds_q_jets, axs[0], "permutation", label="Jets", marker="o", color="tab:green")
plot(ds_q_qm9, axs[0], "permutation", label="QM9", marker="v", color="tab:orange")
axs[0].legend()
axs[0].set_xscale("log")
axs[0].set_ylim([0, 1.0])
axs[0].set_xlim([0, 18000])
axs[0].set_title("Permutation loss")
axs[0].xaxis.set_major_formatter(mticker.ScalarFormatter())
axs[0].set_xticks(xticks)
axs[0].set_xticklabels(xticks, rotation=-45)
plot(ds_B_dist_gecko, axs[1], "versus", label="Gecko", marker="P", color="tab:blue")
plot(
    ds_B_dist_gecko_p,
    axs[1],
    "versus",
    label="Gecko (baseline)",
    color="tab:blue",
    marker="P",
    linestyle="--",
)
plot(ds_B_dist_jets, axs[1], "versus", label="Jets", marker="o", color="tab:green")
plot(
    ds_B_dist_jets_p,
    axs[1],
    "versus",
    label="Jets (baseline)",
    color="tab:green",
    marker="o",
    linestyle="--",
)
plot(ds_B_dist_qm9, axs[1], "versus", label="QM9", marker="v", color="tab:orange")
plot(
    ds_B_dist_qm9_p,
    axs[1],
    "versus",
    label="QM9 (baseline)",
    color="tab:orange",
    marker="v",
    linestyle="--",
)
axs[1].legend()
axs[1].set_xscale("log")
axs[1].set_xlim([0, 18000])
axs[1].set_ylim([0, 1.0])
axs[1].set_title("Local model stability")
axs[1].xaxis.set_major_formatter(mticker.ScalarFormatter())
axs[1].set_xticks(xticks)
axs[1].set_xticklabels(xticks, rotation=-45)
plot(ds_e_dist_gecko, axs[2], "Half", label="Gecko", marker="P", color="tab:blue")
plot(
    ds_e_dist_gecko_p,
    axs[2],
    "Half",
    label="Gecko (baseline)",
    color="tab:blue",
    marker="P",
    linestyle="--",
)
plot(ds_e_dist_jets, axs[2], "Half", label="Jets", marker="o", color="tab:green")
plot(
    ds_e_dist_jets_p,
    axs[2],
    "Half",
    label="Jets (baseline)",
    color="tab:green",
    marker="o",
    linestyle="--",
)
plot(ds_e_dist_qm9, axs[2], "Half", label="QM9", marker="v", color="tab:orange")
plot(
    ds_e_dist_qm9_p,
    axs[2],
    "Half",
    label="QM9 (baseline)",
    color="tab:orange",
    marker="v",
    linestyle="--",
)
axs[2].legend()
axs[2].set_xscale("log")
axs[2].set_ylim([0, 1.0])
axs[2].set_xlim([0, 18000])
axs[2].set_title("Neighbourhood stability")
axs[2].xaxis.set_major_formatter(mticker.ScalarFormatter())
axs[2].set_xticks(xticks)
axs[2].set_xticklabels(xticks, rotation=-45)
plt.tight_layout()
plt.savefig(plot_dir / "stability_measures.eps", format="eps", dpi=150)
plt.show()
