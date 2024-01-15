"""This script reads the DR comparison results and produces a LaTeX table similar to the
one in the paper."""
import pandas as pd
import os
from pathlib import Path

# pick result directories
curr_path = Path(__file__)
gecko_dir = curr_path.parent / "results/gecko/2023-10-17/"
jets_dir = curr_path.parent / "results/jets/2023-10-18/"
qm9_dir = curr_path.parent / "results/qm9/2023-10-18/"


def process_model_column(odf):
    def get_model_type(m):
        if "slisemap" in m:
            return "slisemap"
        elif "PCA" in m:
            return "PCA"
        elif "TSNE" in m:
            return "t-SNE"
        elif "UMAP" in m:
            return "UMAP"
        else:
            raise ValueError(f"Couldn't match m")

    df = odf.copy()
    df.loc[:, "model_type"] = df["model"].apply(get_model_type)
    df.loc[:, "hyperparams"] = (
        df["model"].str.split("random_state_").str[-1].str[8:].str.split(".").str[0]
    )
    df.loc[df["model_type"] == "slisemap", "hyperparams"] = "slisemap"
    return df


def mean_and_std_col(key, means, stds):
    mean = means.loc[key]
    std = stds.loc[key]
    out = mean.copy()
    for idx in out.index:
        if out[idx] < 0.01:
            # convert to scientific notation
            mantissa, exponent = f"{out[idx]:.1E}".split("E")
            exponent = str(int(exponent))
            mean_string = mantissa + " \\cdot 10^{" + exponent + "}"
        else:
            mean_string = f"{out[idx]:.2f}"
        # if std[idx] < 0.01:   # stds in scientific notation is a bit messy
        if False:
            s_mantissa, s_exponent = f"{std[idx]:.1E}".split("E")
            s_exponent = str(int(s_exponent))
            s_string = s_mantissa + " \\cdot 10^{" + s_exponent + "}"
        else:
            s_string = f"{std[idx]:.2f}"
        out[idx] = "$" + mean_string + " \\pm " + s_string + "$"
    return out


def produce_result_table(result_dir, keys, full_table=False):
    fnames = os.listdir(result_dir)
    dfs = []
    for f in fnames:
        df = pd.read_pickle(result_dir / f)
        dfs.append(df)
    df = pd.concat(dfs)
    df = process_model_column(df)
    means = df.groupby(["model_type", "hyperparams"])[
        ["fidelity", "fidelity-nn", "coverage-nn"]
    ].mean()
    stds = df.groupby(["model_type", "hyperparams"])[
        ["fidelity", "fidelity-nn", "coverage-nn"]
    ].std()
    master_df = pd.DataFrame(
        index=["{\sc slisemap}", "PCA", "t-SNE", "UMAP"], columns=means.columns
    )
    means_small = master_df.copy(deep=True)
    for idx, key in keys.items():
        master_df.loc[idx] = mean_and_std_col(key, means, stds)
        means_small.loc[idx] = means.loc[key]
    means_small.reset_index(drop=True, inplace=True)
    best_local = means_small["fidelity"].astype(float).argmin()
    master_df.iloc[best_local][
        "fidelity"
    ] = f"$\\mathbf{{{master_df.iloc[best_local]['fidelity'].replace('$', '')}}}$"
    best_local_nn = means_small["fidelity-nn"].astype(float).argmin()
    master_df.iloc[best_local_nn][
        "fidelity-nn"
    ] = f"$\\mathbf{{{master_df.iloc[best_local]['fidelity-nn'].replace('$', '')}}}$"
    best_coverage = means_small["coverage-nn"].astype(float).argmax()
    master_df.iloc[best_coverage][
        "coverage-nn"
    ] = f"$\\mathbf{{{master_df.iloc[best_local]['coverage-nn'].replace('$', '')}}}$"
    master_df.columns = ["Local loss", "NN Local loss", "NN Coverage"]
    if full_table:
        return master_df, df
    return master_df


# pick hyperparameters to consider
keys = {
    "{\\sc slisemap}": ("slisemap", "slisemap"),
    "PCA": ("PCA", "n_components_2"),
    "t-SNE": ("t-SNE", "perplexity_30"),
    "UMAP": ("UMAP", "n_neighbors_15"),
}

md_g = produce_result_table(gecko_dir, keys)
md_g = md_g.reset_index(names="Model")
md_j = produce_result_table(jets_dir, keys)
md_j = md_j.reset_index(names="Model")
md_q = produce_result_table(qm9_dir, keys)
md_q = md_q.reset_index(names="Model")
master = pd.concat([md_g, md_j, md_q])
master = master.rename(
    columns={
        "Local loss": "Local loss $\\downarrow$",
        "NN Local loss": "NN Local loss $\\downarrow$",
        "NN Coverage": "NN Coverage $\\uparrow$",
    }
)
print(
    master.to_latex(
        index=False,
        bold_rows=False,
        escape=False,
        caption="Comparison of explanation measures for {\sc slismap}, PCA, t-SNE and UMAP for the datasets described in this paper. Bold values indicate best performance.",
        label="tbl:DR_comp",
    )
)
