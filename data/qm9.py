"""
    This scripts downloads and preprocesses the QM9 dataset when called directly.
    The files are placed in the same directory as this script.
"""

import gc
import pickle
import sys
from pathlib import Path
from typing import Callable, Literal, Tuple
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def download(url: str, name: str) -> Path:
    path = Path(__file__).parent / name
    if not path.exists():
        print("Downloading", name)
        urlretrieve(url, path)
    return path


def store(df: pd.DataFrame, name: str, convert: bool = True) -> Path:
    if not "." in name:
        name += ".feather"
    path = Path(__file__).parent / name
    df = df.reset_index(names="index")
    if convert:
        df64 = df.select_dtypes(np.float64)
        df[df64.columns] = df64.astype(np.float32)
        df64 = df.select_dtypes(np.int64)
        df[df64.columns] = df64.astype(np.int32)
    df.to_feather(path, compression="lz4", compression_level=12)
    return path


def exists(name: str) -> bool:
    if not "." in name:
        name += ".feather"
    path = Path(__file__).parent / name
    return path.exists()


def load(name: str) -> pd.DataFrame:
    if not "." in name:
        name += ".feather"
    path = Path(__file__).parent / name
    return pd.read_feather(path).set_index("index")


def check_preprocess(file: str) -> Callable:
    def decorator(function: Callable) -> Callable:
        def check(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"You need to preprocess the data (with `{sys.argv[0]} {file}`) before loading"
                )

        return check

    return decorator


def preprocess(dataset: str, homo_url: str, xyz_url: str, mbtr_url: str):
    if not exists(f"{dataset}_interpretable") or not exists(f"{dataset}_label"):
        import rdkit
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds

        xyz_path = download(xyz_url, f"{dataset}_xyz.xyz")
        print("Parsing XYZ")
        mols = []
        with open(xyz_path, "r") as f:
            while num := f.readline():
                xyz = num + "".join((f.readline() for _ in range(int(num) + 1)))
                mol = Chem.MolFromXYZBlock(xyz)
                mol = Chem.Mol(mol)
                try:
                    rdDetermineBonds.DetermineBonds(mol)
                    Chem.Kekulize(mol)
                    Chem.SanitizeMol(mol)
                    mols.append(mol)
                except Exception as e:
                    mols.append(Chem.Mol())

    if not exists(f"{dataset}_interpretable"):
        import mordred
        import rdkit.RDLogger
        from mordred import descriptors

        print("Calculating interpretable features")
        calc = mordred.Calculator(
            [
                mordred.AtomCount,
                mordred.Weight,
                mordred.BondCount,
                mordred.FragmentComplexity,
                mordred.RotatableBond,
                mordred.HydrogenBond,
                mordred.MoeType.LabuteASA,
                mordred.MomentOfInertia,
                mordred.PBF,
                mordred.Polarizability,
                mordred.TopoPSA,
                mordred.VdwVolumeABC,
                mordred.CarbonTypes,
            ],
            ignore_3D=True,
        )
        rdkit.RDLogger.DisableLog("rdApp.warning")
        interp = calc.pandas(mols)
        interp["vales"] = [Chem.Descriptors.NumValenceElectrons(m) for m in mols]
        rdkit.RDLogger.EnableLog("rdApp.warning")

        print("Filtering molecules")
        interp.fill_missing(np.nan, inplace=True)
        interp.dropna(axis=1, how="all", inplace=True)  # Remove columns with all na
        interp.dropna(axis=0, inplace=True)  # Remove rows with any na
        interp = interp.infer_objects()
        # Drop rare and constant columns
        rare_columns = interp.agg(lambda x: np.mean(x != x.median()) < 0.1)
        interp.drop(columns=interp.columns[rare_columns], inplace=True)
        # Drop heavily correlated columns
        corr_matrix = interp.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = []
        while upper.max().max() > 0.9:
            to_drop.append(upper.columns[upper.max().argmax()])
            upper.drop(index=to_drop[-1:], columns=to_drop[-1:], inplace=True)
        interp.drop(to_drop, axis=1, inplace=True)

        print("Saving interpretable features")
        store(interp, f"{dataset}_interpretable")
        mask = interp.index
        del interp
    else:
        mask = load(f"{dataset}_interpretable").index

    if not exists(f"{dataset}_label"):
        print("Calculating SMILES")
        smiles = []
        for i in mask:
            smiles.append(Chem.MolToSmiles(Chem.RemoveHs(mols[i])))

        homo_path = download(homo_url, f"{dataset}_homo.txt")
        print("Parsing HOMO")
        homo = pd.read_csv(homo_path, names=["homo"]).iloc[mask, :]
        homo.index = smiles

        print("Saving labels")
        store(homo, f"{dataset}_label")
        del smiles, mols, homo

    if not exists(f"{dataset}_mbtr_pca"):
        import scipy

        mbtr_path = download(mbtr_url, f"{dataset}_mbtr.npz")
        print("Parsing MBTR")
        with np.load(mbtr_path) as f:
            mbtr = scipy.sparse.csr_matrix(
                (f["data"], f["indices"], f["indptr"]),
                shape=f["shape"],
                dtype=np.float32,
            )[mask]

        print("Caluclating PCA")
        mbtr = mbtr.toarray()
        gc.collect()
        mbtr = PCA(512).fit_transform(mbtr)
        gc.collect()

        print("Saving MBTR PCA")
        mbtr = pd.DataFrame(mbtr, columns=[f"PCA_{i}" for i in range(mbtr.shape[1])])
        mbtr.index = mask
        store(mbtr, f"{dataset}_mbtr_pca")

    train_model()
    if not exists("qm9_nn"):
        store(load_qm9(0, "nn")[1].reset_index(drop=True), "qm9_nn")
    print(f"Dataset {dataset} fully preprocessed!")


def preprocess_qm9():
    preprocess(
        "qm9",
        "https://zenodo.org/record/4035918/files/HOMO.txt",
        "https://zenodo.org/record/4035918/files/data.xyz",
        "https://zenodo.org/record/4035918/files/mbtr_0.1.npz",
    )


def train_model():
    rf_path = Path(__file__).parent / "qm9_nn.pkl"
    if not rf_path.exists():
        print("Training neural network")
        pipeline = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                (128, 64, 32, 16), verbose=True, random_state=42, early_stopping=True
            ),
        )
        pipeline = TransformedTargetRegressor(pipeline, transformer=StandardScaler())
        X = load("qm9_interpretable")
        y = load("qm9_label")
        pipeline.fit(X, y)
        with open(rf_path, "wb") as f:
            pickle.dump(pipeline, f, protocol=5)


@check_preprocess(__file__)
def load_model(model: Literal["nn"] = "nn") -> MLPRegressor:
    """Load a particle jet classifier.

    Args:
        model: The model to load. Defaults to "rf".

    Returns:
        Sklearn classifier.
    """
    if model == "nn":
        with open(Path(__file__).parent / "qm9_nn.pkl", "rb") as f:
            return pickle.load(f)
    else:
        raise NotImplementedError()


@check_preprocess(__file__)
def load_qm9(
    pca: int = 0, model: Literal[None, "nn"] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the QM9 dataset.

    NOTE: The data has be downloaded first (by calling this script)

    Args:
        pca: Number of PCA features. If zero then interpretable features are returned instead. Defaults to 0.

    Returns:
        X: Features.
        y: Targets.
    """
    y = load("qm9_label")
    if pca <= 0:
        X = load("qm9_interpretable")
    else:
        X = load("qm9_mbtr_pca").iloc[:, :pca]
    if model == "nn" and pca <= 0:
        if exists("qm9_nn"):
            y["homo"] = load("qm9_nn")["homo"].to_numpy()
        else:
            y["homo"] = load_model(model).predict(X)
    elif model is not None:
        raise NotImplementedError()
    return X, y


if __name__ == "__main__":
    preprocess_qm9()
