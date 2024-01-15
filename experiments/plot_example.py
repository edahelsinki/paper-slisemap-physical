import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


from slisemap import Slisemap

# Toy Example
x = np.linspace(-3, 3, 50)
y = np.cos(x)
sm = Slisemap(x[:, None], y, lasso=0, random_state=42)
sm.optimise()
Z = sm.get_Z()
clusters, _ = sm.get_model_clusters(2)
B = sm.get_B()


sns.set_theme(
    context={k: v * 0.6 for k, v in sns.plotting_context("paper").items()},
    style=sns.axes_style("ticks"),
    palette="bright",
    rc={
        "figure.figsize": (4.0, 1.5),
        "savefig.bbox": "tight",
        "figure.dpi": 600,
        "savefig.pad_inches": 1e-4,
    },
)

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=False, sharey=False)
sns.scatterplot(x=x, y=y, hue=clusters, style=clusters, legend=None, ax=ax1)
sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=clusters, style=clusters, legend=None, ax=ax2)
for i in range(sm.n):
    if B[i, 0] < 0:
        ax1.axline((0, B[i, 1]), slope=B[i, 0], color="orange")
    else:
        ax1.axline((0, B[i, 1]), slope=B[i, 0])
ax1.set_xlabel("")
ax1.set_ylabel("f(x)")
ax2.set_ylabel("Embedding")
ax2.axis("equal")
ax2.set_xlabel("")
sns.despine(fig=fig)
plt.tight_layout()
plt.savefig("Fig1.eps")
plt.close()
