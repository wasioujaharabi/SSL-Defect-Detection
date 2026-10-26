import glob, os, sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

run = sys.argv[1]                                   # e.g. runs/full_278640
fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for i, s in enumerate(["shapes", "cutpaste", "perlin"]):
    files = glob.glob(os.path.join(run, "*", "seed*", f"loss_{s}.csv"))
    if not files:
        continue
    g = pd.concat([pd.read_csv(f) for f in files]).groupby("epoch")
    for k, col in enumerate(["train_loss", "val_loss"]):
        m, sd = g[col].mean(), g[col].std().fillna(0)
        ax[k].plot(m.index, m, color=f"C{i}", label=f"{s} (n={len(files)})")
        ax[k].fill_between(m.index, m - sd, m + sd, color=f"C{i}", alpha=0.2)
for k, t in enumerate(["train loss", "val loss (synthetic)"]):
    ax[k].set_title(t); ax[k].set_xlabel("epoch"); ax[k].grid(alpha=0.3)
ax[0].set_ylabel("BCE + Dice"); ax[0].legend()
fig.suptitle("Mean loss over all categories and seeds (band = ±1 std)")
fig.tight_layout()
fig.savefig(os.path.join(run, "loss_mean.png"), dpi=300)
print("saved", os.path.join(run, "loss_mean.png"))