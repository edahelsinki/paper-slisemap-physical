import subprocess
from pathlib import Path

import seaborn as sns
from matplotlib import pyplot as plt
from slisemap import Slisemap

sns.set_theme(
    context={k: v * 0.6 for k, v in sns.plotting_context("paper").items()},
    style=sns.axes_style("ticks"),
    palette="bright",
    rc={
        "figure.figsize": (5.5, 3.0),
        "figure.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 1e-3,
    },
)

MODELS_DIR = Path(__file__).parent.parent / "models"

sm = Slisemap.load(MODELS_DIR / "geckoq_32k_expf.sm", device="cpu")
sm.metadata.set_dimensions(["", ""])
fig = sm.plot(clusters=7, bars=True, figsize=(5.5, 3.0), show=False)
fig.axes[1].set_xlabel("")
plt.savefig("Fig3.pdf")
plt.close()
try:  # Matplotlib does not compress rasterised parts when targetting eps
    subprocess.run(["pdftops", "-rasterize", "never", "-eps", "Fig3.pdf"])
except:
    pass

sm = Slisemap.load(MODELS_DIR / "jets_rf_10k_0.0001_0.0001.sm", device="cpu")
sm.metadata.set_dimensions(["", ""])
fig = sm.plot(clusters=5, bars=True, figsize=(5.5, 2.7), show=False)
fig.axes[1].set_xlabel("")
plt.savefig("Fig5.pdf")
plt.close()
try:
    subprocess.run(["pdftops", "-rasterize", "never", "-eps", "Fig5.pdf"])
except:
    pass

sm = Slisemap.load(MODELS_DIR / "qm9_nn_10k_x_35_0.01_0.001.sm", device="cpu")
sm.metadata.set_dimensions(["", ""])
fig = sm.plot(clusters=5, bars=10, figsize=(5.5, 2.7), show=False)
fig.axes[1].set_xlabel("")
plt.savefig("Fig6.pdf")
plt.close()
try:
    subprocess.run(["pdftops", "-rasterize", "never", "-eps", "Fig6.pdf"])
except:
    pass
