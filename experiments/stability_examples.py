import sys
import numpy as np

sys.path.insert(0, "../../notes/")

from slisemap import Slisemap
import matplotlib.pyplot as plt
import seaborn as sns
from comp_tools import B_distance
from collections.abc import Iterable

np.random.seed(168)

# permutation stability plot

X0 = np.linspace(-3, 3, 41)
Y0 = np.cos(X0)
fig, ax = plt.subplots()
random_indices = np.random.permutation(len(X0))
Xs = X0[random_indices]
ys = Y0[random_indices]
X1 = Xs
y1 = ys
X2 = Xs
y2 = ys
sigma = 0.0
ys1 = y1 + np.random.normal(scale=sigma, size=y1.shape)
ys2 = y2 + np.random.normal(scale=sigma, size=y2.shape)
ys2 = ys2[np.random.permutation(len(ys2))]

# train slisemaps
sm1 = Slisemap(X1[:, None], ys1, lasso=0)
sm1.optimise()
sm2 = Slisemap(X2[:, None], ys2, lasso=0)
sm2.optimise()

# plotting
figsize_mult = 0.8
sns.set_theme(
    context={k: v * 1.0 for k, v in sns.plotting_context("paper").items()},
    style=sns.axes_style("ticks"),
    palette="bright",
    rc={
        "figure.figsize": (4 * figsize_mult, 3.1 * figsize_mult),
        "savefig.bbox": "tight",
        "savefig.pad_inches": 1e-4,
    },
)
fig, ax = plt.subplots(ncols=1)
if not isinstance(ax, Iterable):
    ax = [ax]
sns.scatterplot(
    x=X1, y=ys1, ax=ax[0], color="blue", marker="x", label="Samples $A$", linewidths=1
)
B1 = sm1.get_B()
for i, b in enumerate(B1):
    sns.lineplot(
        x=X1,
        y=X1 * b[0] + b[1],
        ax=ax[0],
        color="blue",
        alpha=1,
        label="Local models $A$" if i == 0 else "_nolegend_",
    )
ax[0].set_ylim([-1.2, 1.4])
sns.scatterplot(
    x=X2,
    y=ys2,
    ax=ax[0],
    c="None",
    edgecolors="orange",
    marker="o",
    label="Samples $B$",
)
B2 = sm2.get_B()
for i, b in enumerate(B2):
    sns.lineplot(
        x=X2,
        y=X2 * b[0] + b[1],
        ax=ax[0],
        color="orange",
        alpha=1,
        label="Local models $B$" if i == 0 else "_nolegend_",
    )
leg = ax[0].legend()
for lh in leg.legend_handles:
    lh.set_alpha(1.0)
ax[0].get_legend().remove()
ax[0].set_ylabel("f(x)")
ax[0].set_yticks([-1, 0, 1])
ax[0].set_xticks([-2, 0, 2])
sns.despine(fig=fig)
fig.savefig("permutation_example.pdf")

# Local model stability
# regenerate data
X0 = np.linspace(-3, 3, 82)
Y0 = np.cos(X0)
X1 = X0[::2]
y1 = Y0[::2]
X2 = X0[1::2]
y2 = Y0[1::2]
sigma = 0.1
ys1 = y1 + np.random.normal(scale=sigma, size=y1.shape)
ys2 = y2 + np.random.normal(scale=sigma, size=y2.shape)

# train
sm1 = Slisemap(X1[:, None], ys1, lasso=0)
sm1.optimise()
sm2 = Slisemap(X2[:, None], ys2, lasso=0)
sm2.optimise()
print(f"Neighbourhood distance: {1 - B_distance(sm1, sm2, match_by_model=True)}")

# produce plot
figsize_mult = 0.8
sns.set_theme(
    context={k: v * 1.0 for k, v in sns.plotting_context("paper").items()},
    style=sns.axes_style("ticks"),
    palette="bright",
    rc={
        "figure.figsize": (4 * figsize_mult, 3 * figsize_mult),
        "savefig.bbox": "tight",
        "savefig.pad_inches": 1e-4,
    },
)
fig, ax = plt.subplots(ncols=1)
if not isinstance(ax, Iterable):
    ax = [ax]
sns.scatterplot(
    x=X1, y=ys1, ax=ax[0], color="blue", marker="x", label="Samples $A$", linewidths=1
)
B1 = sm1.get_B()
ax[0].set_ylim([-1.2, 1.4])
sns.scatterplot(
    x=X2,
    y=ys2,
    ax=ax[0],
    c="None",
    edgecolors="orange",
    marker="o",
    label="Samples $B$",
    linewidths=1,
)
B2 = sm2.get_B()
for i in range(B1.shape[0]):
    b1 = B1[i, :]
    b2 = B2[i, :]
    sns.lineplot(
        x=X1,
        y=X1 * b1[0] + b1[1],
        ax=ax[0],
        color="blue",
        alpha=0.5,
        label="Local models $A$" if i == 0 else "_nolegend_",
    )
    sns.lineplot(
        x=X2,
        y=X2 * b2[0] + b2[1],
        ax=ax[0],
        color="orange",
        alpha=0.5,
        label="Local models $B$" if i == 0 else "_nolegend_",
    )
leg = ax[0].legend()
for lh in leg.legend_handles:
    lh.set_alpha(1.0)
ax[0].get_legend().remove()
ax[0].set_yticks([-1, 0, 1])
ax[0].set_xticks([-2, 0, 2])
sns.despine(fig=fig)
fig.savefig("local_model_stability_example.pdf")

# Neighbourhood stability
X0 = np.linspace(-3, 3, 41)
Y0 = np.cos(X0)
Xs = X0
ys = Y0
sm = Slisemap(Xs[:, None], ys, lasso=0)
sm.optimise()

# switches for extra plotting options
plot_function = True
plot_embedding = False
sns.set_theme(
    context={k: v * 1 for k, v in sns.plotting_context("paper").items()},
    style=sns.axes_style("ticks"),
    palette="bright",
    rc={
        "figure.figsize": (6, 3)
        if plot_embedding
        else (4 * figsize_mult, 3 * figsize_mult),
        "savefig.bbox": "tight",
        "savefig.pad_inches": 1e-4,
    },
)
colors = sns.color_palette("bright", 10)
Z = sm.get_Z(rotate=True)
middle = len(Xs) // 2
fig = plt.figure()
if plot_function and plot_embedding:
    gs = fig.add_gridspec(2, 3)
    ax1 = fig.add_subplot(gs[:, 0:2])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 2])
    ax = [ax1, ax2, ax3]
elif plot_function:
    ax1 = fig.add_subplot()
    ax = [ax1]
elif plot_embedding:
    gs = fig.add_gridspec(2, 3)
    ax1 = fig.add_subplot(gs[:, 0:2])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 2])
    ax = [ax1, ax2, ax3]
if plot_function:
    sns.scatterplot(
        x=Xs[:middle],
        y=ys[:middle],
        ax=ax[0],
        color=colors[0],
        marker="x",
        linewidths=1,
    )
    sns.scatterplot(
        x=Xs[middle + 1 :],
        y=ys[middle + 1 :],
        ax=ax[0],
        color="None",
        edgecolor=colors[1],
        linewidths=1,
    )
    for i, b in enumerate(sm.get_B()):
        color = "blue" if b[0] > 0 else "orange"
        sns.lineplot(x=Xs, y=Xs * b[0] + b[1], ax=ax[0], color=color, alpha=1)
    sns.scatterplot(
        x=[Xs[middle]],
        y=[ys[middle]],
        ax=ax[0],
        marker="s",
        color=colors[3],
        edgecolor="w",
        linewidths=1.5,
        s=100,
        zorder=100,
    )
    ax[0].set_ylim([-1.2, 1.4])
    ax[0].set_yticks([-1, 0, 1])
    ax[0].set_xticks([-2, 0, 2])
if plot_embedding:
    sns.scatterplot(
        x=Z[:middle, 0],
        y=Z[:middle, 1],
        ax=ax[2],
        color="blue",
        marker="x",
        linewidths=1,
    )
    sns.scatterplot(
        x=[-Z[middle, 0]], y=[Z[middle, 1]], ax=ax[2], color="green", marker="*", s=300
    )
    sns.scatterplot(
        x=Z[middle + 1 :, 0],
        y=Z[middle + 1 :, 1],
        ax=ax[2],
        color="None",
        edgecolor="orange",
        linewidths=1,
    )
    sns.scatterplot(
        x=Z[:middle, 0],
        y=Z[:middle, 1],
        ax=ax[1],
        color="blue",
        marker="x",
        linewidths=1,
    )
    sns.scatterplot(
        x=[Z[middle, 0]], y=[Z[middle, 1]], ax=ax[1], color="green", marker="*", s=300
    )
    sns.scatterplot(
        x=Z[middle + 1 :, 0],
        y=Z[middle + 1 :, 1],
        ax=ax[1],
        color="None",
        edgecolor="orange",
        linewidths=1,
    )
    ax[1].set_xlabel("SLISEMAP 1")
    ax[1].set_ylabel("SLISEMAP 2")
    ax[2].set_xlabel("SLISEMAP 1")
    ax[2].set_ylabel("SLISEMAP 2")
    ax[1].set_ylim([-4, 4])
    ax[2].set_ylim([-4, 4])
fig.tight_layout()
sns.despine(fig=fig)
fig.savefig("E_distance_example.pdf")
