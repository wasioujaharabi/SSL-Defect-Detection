#!/usr/bin/env python
"""
Synthetic-anomaly defect detection on MVTec AD.

Question: which synthetic defect strategy best teaches a model to find REAL defects?

Methods:
  shapes, cutpaste, perlin  synthetic defect strategies. For each one, a U-Net with an ImageNet-pretrained
                            ResNet18 encoder is fine-tuned end to end to segment the synthetic defects.
  knn                       training-free baseline (PatchCore-style): frozen ResNet18 patch features,
                            anomaly score = distance to the nearest normal training patch.
Evaluation on the real MVTec test set: image AUROC, pixel AUROC, pixel AP, image AUROC per defect type.
Every method runs once per seed in --seeds; the summary reports mean +- std over seeds.

Outputs in --out_dir:
  results.csv                      one row per (category, method, seed)
  summary.csv                      mean and std over seeds per (category, method)
  summary_image_auroc.csv / .png   image AUROC "mean +- std" table and bar chart
  per_defect_type_summary.csv      image AUROC per real defect type, mean and std over seeds
  <category>/samples/              synthetic defect examples for the poster (grid PNG + PDF, single images)
  <category>/real_defects.png      one real defect per type
  <category>/seed<S>/              loss_<method>.png/.csv (redrawn every epoch), loss_all.png, metrics.png,
                                   qualitative.png, per_defect_type.csv, <method>.pt checkpoints

Usage:
  python SSL_defect_detection.py --prepare --dtd_root <dir>              # download ResNet18 weights + DTD
  python SSL_defect_detection.py --data_root <dir> --samples_only        # only the poster sample figures
  python SSL_defect_detection.py --data_root <dir> --categories carpet bottle screw --seeds 0 1 2
"""
import argparse
import glob
import os
import random
import time
from datetime import datetime
from scipy import ndimage
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display on the cluster
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import roc_auc_score, average_precision_score

STRATEGIES = ["shapes", "cutpaste", "perlin"]           # synthetic defect generators -> trained models
BASELINES = ["knn"]                                     # training-free
METHODS = STRATEGIES + BASELINES
ENCODER_PARTS = ("stem", "l1", "l2", "l3", "l4")        # SegNet modules that come from the pretrained ResNet18
OBJECTS = {"bottle", "cable", "capsule", "hazelnut", "metal_nut", "pill", "screw", "toothbrush", "transistor", "zipper"}
NO_FLIP = {"metal_nut", "transistor", "screw"}  # a flipped part is a real defect type here, so no flip augmentation
NAN = float("nan")


def log(*a):
    print(*a, flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", default=None, help="folder with unpacked MVTec categories (carpet/, bottle/, ...)")
    p.add_argument("--out_dir", default="runs")
    p.add_argument("--categories", nargs="+", default=["carpet", "bottle", "screw"])
    p.add_argument("--strategies", nargs="+", default=METHODS, choices=METHODS,
                   help="methods to run: synthetic strategies and/or the knn baseline")
    p.add_argument("--seeds", nargs="+", type=int, default=[0], help="one full run per seed, e.g. --seeds 0 1 2")
    p.add_argument("--img_size", type=int, default=256, help="training size, must be divisible by 32")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4, help="decoder learning rate")
    p.add_argument("--enc_lr_mult", type=float, default=0.1, help="encoder lr = lr * this (1.0 = same lr everywhere)")
    p.add_argument("--anomaly_p", type=float, default=0.5, help="share of training images that get a defect")
    p.add_argument("--val_frac", type=float, default=0.1, help="normal train images held out for the val loss")
    p.add_argument("--dtd_root", default="~/data/dtd", help="where DTD textures are (or get downloaded by --prepare)")
    p.add_argument("--no_dtd", action="store_true", help="Perlin textures from training images instead of DTD")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--n_samples", type=int, default=4, help="examples per strategy in the poster figure")
    p.add_argument("--sample_size", type=int, default=512, help="resolution of poster samples, divisible by 32")
    p.add_argument("--save_all_ckpts", action="store_true", help="save checkpoints for every seed (default: first only)")
    p.add_argument("--samples_only", action="store_true", help="only save sample figures, no training")
    p.add_argument("--prepare", action="store_true", help="download ResNet18 weights and DTD, then exit")
    a = p.parse_args()
    if not a.prepare and a.data_root is None:
        p.error("--data_root is required (unless --prepare)")
    for name in ["img_size", "sample_size"]:
        if getattr(a, name) % 32:
            p.error(f"--{name} must be divisible by 32")
    a.dtd_root = os.path.expanduser(a.dtd_root)
    return a


# =====================================================================================
# Data
# =====================================================================================
def load_img(path, size):
    im = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def load_mask(path, size):
    m = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    return (np.asarray(m) > 127).astype(np.float32)


def mask_path(cat_dir, defect_type, img_path):
    name = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(cat_dir, "ground_truth", defect_type, f"{name}_mask.png")


def load_category(data_root, cat, size):
    cat_dir = os.path.join(os.path.expanduser(data_root), cat)
    if not os.path.isdir(cat_dir):
        raise FileNotFoundError(f"{cat_dir} not found. Unpack MVTec first so that <data_root>/{cat}/train exists")
    train_paths = sorted(glob.glob(os.path.join(cat_dir, "train", "good", "*.png")))
    test_paths, test_types = [], []
    for t in sorted(os.listdir(os.path.join(cat_dir, "test"))):
        for pth in sorted(glob.glob(os.path.join(cat_dir, "test", t, "*.png"))):
            test_paths.append(pth)
            test_types.append(t)
    test_types = np.array(test_types)
    masks = [np.zeros((size, size), np.float32) if t == "good" else load_mask(mask_path(cat_dir, t, pth), size)
             for pth, t in zip(test_paths, test_types)]
    d = dict(cat_dir=cat_dir, train_paths=train_paths, test_paths=test_paths, test_types=test_types,
             train=np.stack([load_img(pth, size) for pth in train_paths]),
             test=np.stack([load_img(pth, size) for pth in test_paths]),
             masks=np.stack(masks), labels=(test_types != "good").astype(int),
             defect_types=sorted({str(t) for t in test_types} - {"good"}))
    log(f"[{cat}] train {len(d['train'])} | test {len(d['test'])} ({d['labels'].sum()} defective) | "
        f"defect types: {d['defect_types']}")
    return d


def load_dtd(args, size):
    """300 random DTD textures (as in DRAEM), or None if DTD is not available. Same bank for every seed."""
    if args.no_dtd:
        return None
    try:
        from torchvision.datasets import DTD
        dtd = DTD(args.dtd_root, split="train", download=False)
        idx = np.random.default_rng(0).choice(len(dtd), size=min(300, len(dtd)), replace=False)
        bank = np.stack([np.asarray(dtd[i][0].convert("RGB").resize((size, size), Image.BILINEAR), np.float32) / 255.0
                         for i in idx])
        log(f"DTD textures loaded: {bank.shape}")
        return bank
    except Exception as e:
        log(f"WARNING: DTD not available at {args.dtd_root} ({e}).\n"
            f"         Perlin will use training images as textures. Run with --prepare on the login node first.")
        return None


# =====================================================================================
# Synthetic anomaly generators
# Each takes a normal image (H, W, 3) float32 in [0, 1] and returns (defective_image, mask (H, W) in {0, 1}).
# They use np.random, so seeding np.random makes them reproducible.
# =====================================================================================
def shapes_anomaly(img):
    """Simplest strategy. 1-3 random shapes (ellipse, polygon or scratch-like line) filled with a random
    colour plus pixel noise, partly blended with the original. No knowledge of the object or texture."""
    H, W, _ = img.shape
    m = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(m)
    for _ in range(np.random.randint(1, 4)):
        cx, cy = int(np.random.randint(0, W)), int(np.random.randint(0, H))
        r = int(np.random.randint(H // 40, H // 8))
        kind = np.random.choice(["ellipse", "polygon", "line"])
        if kind == "ellipse":
            rx, ry = r, max(2, int(r * np.random.uniform(0.3, 1.0)))
            d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
        elif kind == "polygon":
            pts = [(cx + int(np.random.randint(-r, r)), cy + int(np.random.randint(-r, r)))
                   for _ in range(np.random.randint(3, 7))]
            d.polygon(pts, fill=255)
        else:
            dx, dy = int(np.random.randint(-3 * r, 3 * r)), int(np.random.randint(-3 * r, 3 * r))
            d.line([(cx, cy), (cx + dx, cy + dy)], fill=255, width=max(2, int(np.random.randint(2, 7) * H / 256)))
    mask = (np.asarray(m) > 0).astype(np.float32)
    if mask.sum() == 0:
        mask[H // 2 - H // 32:H // 2 + H // 32, W // 2 - W // 32:W // 2 + W // 32] = 1
    fill = np.clip(np.random.rand(3) + np.random.normal(0, 0.1, img.shape), 0, 1)
    beta = np.random.uniform(0.0, 0.5)                       # how much of the original stays visible
    mm = mask[..., None]
    out = img * (1 - mm) + (beta * img + (1 - beta) * fill) * mm
    return out.astype(np.float32), mask


def cutpaste_anomaly(img):
    """CutPaste (Li et al., CVPR 2021). A rectangle (2-15% of the image, random aspect ratio) is cut
    from the image itself, colour-jittered, maybe flipped, and pasted at another position.
    The defect uses the object's own colours/texture, so it is subtle and structural."""
    H, W, _ = img.shape
    area = np.random.uniform(0.02, 0.15) * H * W
    aspect = np.exp(np.random.uniform(np.log(0.3), np.log(3.3)))
    h = int(np.clip(round(np.sqrt(area * aspect)), 4, H - 1))
    w = int(np.clip(round(np.sqrt(area / aspect)), 4, W - 1))
    y0, x0 = np.random.randint(0, H - h), np.random.randint(0, W - w)
    y1, x1 = np.random.randint(0, H - h), np.random.randint(0, W - w)
    patch = img[y0:y0 + h, x0:x0 + w].copy()
    patch = np.clip(patch * np.random.uniform(0.8, 1.2, 3) + np.random.uniform(-0.1, 0.1), 0, 1)
    if np.random.rand() < 0.5:
        patch = patch[:, ::-1]
    out = img.copy()
    out[y1:y1 + h, x1:x1 + w] = patch
    mask = np.zeros((H, W), np.float32)
    mask[y1:y1 + h, x1:x1 + w] = 1
    return out.astype(np.float32), mask


def rand_perlin_2d(shape, res):
    """2D Perlin noise in roughly [-1, 1]; shape must be divisible by res."""
    fade = lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    grads = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    tile = lambda s1, s2: np.repeat(np.repeat(grads[s1[0]:s1[1], s2[0]:s2[1]], d[0], 0), d[1], 1)
    g = grid[:shape[0], :shape[1]]
    dot = lambda gr, sh: (np.stack((g[..., 0] + sh[0], g[..., 1] + sh[1]), -1) * gr[:shape[0], :shape[1]]).sum(-1)
    n00 = dot(tile([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile([0, -1], [1, None]), [0, -1])
    n11 = dot(tile([1, None], [1, None]), [-1, -1])
    t = fade(g)
    lerp = lambda a, b, w: a + w * (b - a)
    return np.sqrt(2) * lerp(lerp(n00, n10, t[..., 0]), lerp(n01, n11, t[..., 0]), t[..., 1])


class PerlinAnomaly:
    """DRAEM-style (Zavrtanik et al., ICCV 2021). Thresholded Perlin noise gives irregular, blob-like
    regions of random scale; they are filled with a foreign texture (DTD, or another training image),
    partly blended with the original. Produces texture/colour defects with natural shapes."""

    def __init__(self, textures):
        self.textures = textures

    def __call__(self, img):
        H, W, _ = img.shape
        for _ in range(10):                                   # retry until the mask is not (almost) empty
            res = (2 ** np.random.randint(0, 6), 2 ** np.random.randint(0, 6))
            mask = (rand_perlin_2d((H, W), res) > 0.5).astype(np.float32)
            if mask.sum() > 20 * (H * W) / 256 ** 2:
                break
        tex = self.textures[np.random.randint(len(self.textures))]
        if tex.shape[:2] != (H, W):
            tex = np.asarray(Image.fromarray((tex * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR),
                             np.float32) / 255.0
        tex = np.clip(tex * np.random.uniform(0.7, 1.3, 3), 0, 1)
        beta = np.random.uniform(0.0, 0.8)                    # how much of the original stays visible
        m = mask[..., None]
        out = img * (1 - m) + ((1 - beta) * tex + beta * img) * m
        return out.astype(np.float32), mask

#new functions
def foreground_mask(img, thr=0.2):
    """Rough object mask: pixels clearly different from the background colour (median of the border)."""
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    bg = np.median(border, axis=0)
    fg = np.abs(img - bg).max(-1) > thr
    fg = ndimage.binary_opening(fg, iterations=2)
    fg = ndimage.binary_fill_holes(ndimage.binary_closing(fg, iterations=3))
    return fg.astype(np.float32)


class OnObject:
    """Wraps a generator so the defect is only kept where the object is (object categories only)."""

    def __init__(self, gen):
        self.gen = gen

    def __call__(self, img):
        H, W, _ = img.shape
        min_px = 30 * H * W / 256 ** 2
        fg = foreground_mask(img)
        if fg.sum() < min_px:                        # object not found -> plain generator
            return self.gen(img)
        for _ in range(10):                          # retry until the defect overlaps the object
            out, mask = self.gen(img)
            mask = mask * fg
            if mask.sum() >= min_px:
                break
        m = mask[..., None]
        return (img * (1 - m) + out * m).astype(np.float32), mask

# def make_generators(textures):
#     return {"shapes": shapes_anomaly, "cutpaste": cutpaste_anomaly, "perlin": PerlinAnomaly(textures)}

def make_generators(textures, on_object=False):
    gens = {"shapes": shapes_anomaly, "cutpaste": cutpaste_anomaly, "perlin": PerlinAnomaly(textures)}
    return {k: OnObject(g) for k, g in gens.items()} if on_object else gens
# =====================================================================================
# Poster figures
# =====================================================================================
def to_png(arr, path):
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(path)


# def save_synthetic_samples(data, gens, strategies, out_dir, cat, args):
#     """Same normal images for every strategy so the figure compares strategies fairly.
#     Saves samples/synthetic_grid.{png,pdf} plus every single image and mask as PNG."""
#     sd = os.path.join(out_dir, "samples")
#     os.makedirs(sd, exist_ok=True)
#     seed = args.seeds[0]
#     rng = np.random.default_rng(seed)
#     n = min(args.n_samples, len(data["train_paths"]))
#     pick = rng.choice(len(data["train_paths"]), size=n, replace=False)
#     bases = [load_img(data["train_paths"][i], args.sample_size) for i in pick]
#     out = {}
#     for s in strategies:
#         np.random.seed(seed)
#         out[s] = [gens[s](b.copy()) for b in bases]
#         for k, (im, mk) in enumerate(out[s]):
#             to_png(im, f"{sd}/{s}_{k}_image.png")
#             to_png(mk, f"{sd}/{s}_{k}_mask.png")
#     for k, b in enumerate(bases):
#         to_png(b, f"{sd}/normal_{k}.png")

#     cols = ["normal"] + [f"{s}{suffix}" for s in strategies for suffix in ["", " mask"]]
#     fig, ax = plt.subplots(n, len(cols), figsize=(2.2 * len(cols), 2.2 * n), squeeze=False)
#     for r in range(n):
#         ax[r, 0].imshow(bases[r])
#         for j, s in enumerate(strategies):
#             im, mk = out[s][r]
#             ax[r, 1 + 2 * j].imshow(im)
#             ax[r, 2 + 2 * j].imshow(mk, cmap="gray", vmin=0, vmax=1)
#     for c, name in enumerate(cols):
#         ax[0, c].set_title(name, fontsize=12)
#     for a in ax.ravel():
#         a.set_xticks([]); a.set_yticks([])
#     fig.suptitle(f"Synthetic defects - {cat}", fontsize=14)
#     fig.tight_layout()
#     fig.savefig(f"{sd}/synthetic_grid.png", dpi=300)
#     fig.savefig(f"{sd}/synthetic_grid.pdf")
#     plt.close(fig)
#     log(f"[{cat}] synthetic samples -> {sd}/")

def save_poster_grid(args, strategies, dtd, cats=None):
    """ONE poster grid: rows = categories (one normal image each), cols = normal + strategies.
    Same image for all strategies in a row; red outline = defect mask."""
    cats = cats or args.categories
    seed = args.seeds[0]
    rng = np.random.default_rng(seed)
    cols = ["normal"] + strategies
    fig, ax = plt.subplots(len(cats), len(cols), figsize=(2.4 * len(cols), 2.4 * len(cats)), squeeze=False)
    for r, cat in enumerate(cats):
        paths = sorted(glob.glob(os.path.join(os.path.expanduser(args.data_root), cat, "train", "good", "*.png")))
        textures = dtd if dtd is not None else np.stack([load_img(p, args.img_size) for p in paths[:50]])
        gens = make_generators(textures, on_object=cat in OBJECTS)
        base = load_img(paths[rng.integers(len(paths))], args.sample_size)
        ax[r, 0].imshow(base); ax[r, 0].set_ylabel(cat, fontsize=12)
        for j, s in enumerate(strategies):
            np.random.seed(seed * 100 + r)
            im, mk = gens[s](base.copy())
            ax[r, j + 1].imshow(im)
            if mk.any():
                ax[r, j + 1].contour(mk, levels=[0.5], colors="red", linewidths=0.8)
    for c, name in enumerate(cols):
        ax[0, c].set_title(name, fontsize=13)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.subplots_adjust(wspace=0.03, hspace=0.03)
    fig.savefig(os.path.join(args.out_dir, "synthetic_grid.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(args.out_dir, "synthetic_grid.pdf"), bbox_inches="tight")
    plt.close(fig)


def save_real_defects(data, out_dir, cat, size):
    types = data["defect_types"]
    fig, ax = plt.subplots(2, len(types), figsize=(2.6 * len(types), 5.4), squeeze=False)
    for j, t in enumerate(types):
        i = int(np.where(data["test_types"] == t)[0][0])
        pth = data["test_paths"][i]
        ax[0, j].imshow(load_img(pth, size)); ax[0, j].set_title(t, fontsize=12)
        ax[1, j].imshow(load_mask(mask_path(data["cat_dir"], t, pth), size), cmap="gray", vmin=0, vmax=1)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"Real defects - {cat}", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/real_defects.png", dpi=300)
    plt.close(fig)


# =====================================================================================
# Dataset, model, loss
# =====================================================================================
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def worker_init(worker_id):
    np.random.seed(torch.initial_seed() % 2 ** 32)          # different synthetic defects in each worker


class SyntheticAnomalyDataset(Dataset):
    """Normal images; with probability p a synthetic defect is added on the fly (new defect every epoch)."""

    def __init__(self, imgs, generator, p, hflip):
        self.imgs, self.generator, self.p, self.hflip = imgs, generator, p, hflip

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = self.imgs[i]
        if self.hflip and np.random.rand() < 0.5:
            img = img[:, ::-1]
        img = np.ascontiguousarray(img)
        if np.random.rand() < self.p:
            img, mask = self.generator(img)
        else:
            mask = np.zeros(img.shape[:2], np.float32)
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()
        y = torch.from_numpy(np.ascontiguousarray(mask))[None].float()
        return x, y


def make_val_set(imgs, generator, seed):
    """Fixed synthetic validation set (every second image gets a defect), identical in every epoch,
    so the val loss curve is comparable over time. It measures fit to the synthetic task only."""
    state = np.random.get_state()
    np.random.seed(seed + 1000)
    xs, ys = [], []
    for i, img in enumerate(imgs):
        if i % 2 == 0:
            im, mk = generator(img.copy())
        else:
            im, mk = img, np.zeros(img.shape[:2], np.float32)
        xs.append(torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1))
        ys.append(torch.from_numpy(mk)[None])
    np.random.set_state(state)
    return torch.stack(xs).float(), torch.stack(ys).float()


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1))


class SegNet(nn.Module):
    """U-Net with an ImageNet-pretrained ResNet18 encoder, fine-tuned end to end.
    Input: RGB in [0, 1]. Output: 1-channel defect logits at input resolution."""

    def __init__(self):
        super().__init__()
        r = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)           # 1/2,  64 ch
        self.l1 = nn.Sequential(r.maxpool, r.layer1)                # 1/4,  64
        self.l2, self.l3, self.l4 = r.layer2, r.layer3, r.layer4    # 1/8 128, 1/16 256, 1/32 512
        self.up4, self.up3 = UpBlock(512, 256, 256), UpBlock(256, 128, 128)
        self.up2, self.up1 = UpBlock(128, 64, 64), UpBlock(64, 64, 64)
        self.head = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 1, 1))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        size = x.shape[-2:]
        x = (x - self.mean) / self.std
        s0 = self.stem(x); s1 = self.l1(s0); s2 = self.l2(s1); s3 = self.l3(s2); s4 = self.l4(s3)
        d = self.up1(self.up2(self.up3(self.up4(s4, s3), s2), s1), s0)
        d = F.interpolate(d, size=size, mode="bilinear", align_corners=False)
        return self.head(d)


def seg_loss(logits, mask):
    """BCE + soft Dice."""
    bce = F.binary_cross_entropy_with_logits(logits, mask)
    p = torch.sigmoid(logits)
    inter = (p * mask).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + mask.sum((1, 2, 3)) + 1)
    return bce + dice.mean()


# =====================================================================================
# Training with loss monitoring
# =====================================================================================
def plot_loss(hist, path, title):
    ep = hist["epoch"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ep, hist["train_loss"], "o-", ms=3, label="train")
    if any(v is not None for v in hist["val_loss"]):
        ax.plot(ep, hist["val_loss"], "s--", ms=3, label="val (synthetic)")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss (BCE + Dice)"); ax.set_title(title)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):                       # older PyTorch
        return torch.cuda.amp.GradScaler(enabled=enabled)


@torch.no_grad()
def val_loss(model, val_xy, device, use_amp, batch=16):
    model.eval()
    xs, ys = val_xy
    total = 0.0
    for i in range(0, len(xs), batch):
        x, y = xs[i:i + batch].to(device), ys[i:i + batch].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        total += seg_loss(logits.float(), y).item() * len(x)
    return total / len(xs)


def train(strategy, gen, train_imgs, val_imgs, args, seed, hflip, run_dir, cat, device):
    set_seed(seed)
    ds = SyntheticAnomalyDataset(train_imgs, gen, args.anomaly_p, hflip)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers,
                    worker_init_fn=worker_init, persistent_workers=args.num_workers > 0,
                    pin_memory=device.type == "cuda")
    val_xy = make_val_set(val_imgs, gen, seed) if len(val_imgs) else None

    model = SegNet().to(device)
    enc = [p for n, p in model.named_parameters() if n.split(".")[0] in ENCODER_PARTS]
    dec = [p for n, p in model.named_parameters() if n.split(".")[0] not in ENCODER_PARTS]
    enc_lr = args.lr * args.enc_lr_mult
    opt = torch.optim.AdamW([{"params": enc, "lr": enc_lr}, {"params": dec, "lr": args.lr}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(dl))
    use_amp = device.type == "cuda"
    scaler = make_scaler(use_amp)
    log(f"  trainable params: encoder {sum(p.numel() for p in enc) / 1e6:.1f}M (lr {enc_lr:.0e}) | "
        f"decoder {sum(p.numel() for p in dec) / 1e6:.1f}M (lr {args.lr:.0e})")

    hist = {"epoch": [], "train_loss": [], "val_loss": [], "lr": []}
    loss_png, loss_csv = f"{run_dir}/loss_{strategy}.png", f"{run_dir}/loss_{strategy}.csv"
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
            loss = seg_loss(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            total += loss.item()
        tr = total / len(dl)
        va = val_loss(model, val_xy, device, use_amp) if val_xy is not None else None
        hist["epoch"].append(ep); hist["train_loss"].append(tr); hist["val_loss"].append(va)
        hist["lr"].append(opt.param_groups[1]["lr"])
        pd.DataFrame(hist).to_csv(loss_csv, index=False)
        plot_loss(hist, loss_png, f"{cat} / {strategy} / seed {seed}")
        el = time.time() - t0
        va_txt = f" | val {va:.4f}" if va is not None else ""
        log(f"  [{cat}/{strategy}/s{seed}] epoch {ep:3d}/{args.epochs} | train {tr:.4f}{va_txt} | "
            f"lr {hist['lr'][-1]:.1e} | {el:.0f}s, ~{el / ep * (args.epochs - ep) / 60:.1f} min left")
    return model, hist, time.time() - t0


# =====================================================================================
# Training-free baseline: kNN on frozen pretrained patch features (PatchCore-style)
# =====================================================================================
class PatchFeatureExtractor(nn.Module):
    """Frozen ImageNet ResNet18 (same weights the U-Net encoder starts from). Returns locally averaged
    layer2 + layer3 features at 1/8 resolution: (B, 128 + 256 = 384, H/8, W/8)."""

    def __init__(self):
        super().__init__()
        r = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.l1, self.l2, self.l3 = r.layer1, r.layer2, r.layer3
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        x = (x - self.mean) / self.std
        f2 = self.l2(self.l1(self.stem(x)))
        f3 = self.l3(f2)
        f2 = F.avg_pool2d(f2, 3, stride=1, padding=1, count_include_pad=False)
        f3 = F.avg_pool2d(f3, 3, stride=1, padding=1, count_include_pad=False)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f2, f3], 1)


@torch.no_grad()
def extract_features(extractor, imgs, device, batch=32):
    out = []
    for i in range(0, len(imgs), batch):
        x = torch.from_numpy(imgs[i:i + batch]).permute(0, 3, 1, 2).to(device)
        out.append(extractor(x))
    return torch.cat(out)


@torch.no_grad()
def knn_baseline(train_imgs, test_imgs, img_size, device, chunk=1024):
    """Memory bank = all patch features of the normal training images. Each test patch is scored by its
    distance to the nearest bank patch; the patch map is upsampled and smoothed like the U-Net maps."""
    t0 = time.time()
    ext = PatchFeatureExtractor().to(device)
    train_f = extract_features(ext, train_imgs, device)
    C, h, w = train_f.shape[1:]
    bank = train_f.permute(0, 2, 3, 1).reshape(-1, C)
    del train_f
    q = extract_features(ext, test_imgs, device).permute(0, 2, 3, 1).reshape(-1, C)
    dmin = torch.cat([torch.cdist(q[i:i + chunk], bank).min(1).values for i in range(0, len(q), chunk)])
    maps = dmin.view(len(test_imgs), 1, h, w)
    maps = F.interpolate(maps, size=(img_size, img_size), mode="bilinear", align_corners=False)
    maps = TF.gaussian_blur(maps, kernel_size=33, sigma=4)
    secs = time.time() - t0
    log(f"  [knn] memory bank {tuple(bank.shape)} | {secs:.0f}s")
    return maps[:, 0].cpu().numpy(), secs


# =====================================================================================
# Evaluation
# =====================================================================================
@torch.no_grad()
def predict_maps(model, test_imgs, device, batch=16):
    model.eval()
    maps = []
    x_all = torch.from_numpy(test_imgs).permute(0, 3, 1, 2)
    for i in range(0, len(x_all), batch):
        p = torch.sigmoid(model(x_all[i:i + batch].to(device)).float())
        p = TF.gaussian_blur(p, kernel_size=33, sigma=4)
        maps.append(p[:, 0].cpu())
    return torch.cat(maps).numpy()


def evaluate(maps, data):
    scores = maps.reshape(len(maps), -1).max(1)
    gt = data["masks"].ravel().astype(np.uint8)
    res = {"image_auroc": roc_auc_score(data["labels"], scores),
           "pixel_auroc": roc_auc_score(gt, maps.ravel()),
           "pixel_ap": average_precision_score(gt, maps.ravel())}
    for t in data["defect_types"]:                            # each defect type vs. the good test images
        sel = (data["test_types"] == t) | (data["test_types"] == "good")
        res[f"img_auroc_{t}"] = roc_auc_score(data["labels"][sel], scores[sel])
    return res


def save_seed_plots(res, hists, maps, data, methods, run_dir, cat):
    main = ["image_auroc", "pixel_auroc", "pixel_ap"]
    per_type = [c for c in res.columns if c.startswith("img_auroc_")]
    res[per_type].rename(columns=lambda c: c[len("img_auroc_"):]).round(3).to_csv(f"{run_dir}/per_defect_type.csv")

    fig, ax = plt.subplots(figsize=(6, 4))
    res[main].plot.bar(ax=ax, rot=0, ylim=(0, 1))
    ax.axhline(0.5, color="gray", ls="--", lw=1); ax.set_title(f"{cat}: metrics per method"); ax.set_xlabel("")
    ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.tight_layout(); fig.savefig(f"{run_dir}/metrics.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    if hists:
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, s in enumerate(hists):
            c = f"C{i}"
            ax.plot(hists[s]["epoch"], hists[s]["train_loss"], "-", color=c, label=f"{s} train")
            if hists[s]["val_loss"][0] is not None:
                ax.plot(hists[s]["epoch"], hists[s]["val_loss"], "--", color=c, label=f"{s} val")
        ax.set_xlabel("epoch"); ax.set_ylabel("loss (BCE + Dice)"); ax.set_title(f"{cat}: loss curves")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(f"{run_dir}/loss_all.png", dpi=200); plt.close(fig)

    tt = data["test_types"]
    idx = [int(np.where(tt == t)[0][0]) for t in data["defect_types"]] + [int(np.where(tt == "good")[0][0])]
    cols = ["input", "ground truth"] + methods
    fig, ax = plt.subplots(len(idx), len(cols), figsize=(2.4 * len(cols), 2.4 * len(idx)), squeeze=False)
    for r, i in enumerate(idx):
        ax[r, 0].imshow(data["test"][i]); ax[r, 0].set_ylabel(tt[i], fontsize=11)
        ax[r, 1].imshow(data["masks"][i], cmap="gray", vmin=0, vmax=1)
        for c, s in enumerate(methods):
            m = maps[s]
            if s in BASELINES:                                # distances, not probabilities: rescale for display
                m = (m - m.min()) / (m.max() - m.min() + 1e-8)
            ax[r, c + 2].imshow(data["test"][i])
            ax[r, c + 2].imshow(m[i], cmap="jet", alpha=0.5, vmin=0, vmax=1)
    for c, name in enumerate(cols):
        ax[0, c].set_title(name)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(f"{run_dir}/qualitative.png", dpi=200); plt.close(fig)


# =====================================================================================
# Results files and summary over seeds
# =====================================================================================
def append_row(csv_path, row):
    df = pd.DataFrame([row])
    if os.path.exists(csv_path):                              # align columns across categories
        df = pd.concat([pd.read_csv(csv_path), df], ignore_index=True)
    df.to_csv(csv_path, index=False)


def write_summary(rows, methods, out_dir):
    df = pd.DataFrame(rows)
    main = ["image_auroc", "pixel_auroc", "pixel_ap"]
    cats = list(dict.fromkeys(df["category"]))
    methods = [m for m in methods if m in set(df["strategy"])]

    g = df.groupby(["category", "strategy"])[main]
    mean, std = g.mean(), g.std().fillna(0.0)                 # std is NaN with a single seed -> 0
    summ = mean.add_suffix("_mean").join(std.add_suffix("_std"))
    summ["n_seeds"] = g.size()
    summ.round(4).to_csv(os.path.join(out_dir, "summary.csv"))

    per_seed = df.groupby(["seed", "strategy"])["image_auroc"].mean().unstack()   # mean over categories
    table = pd.DataFrame(index=cats + ["mean"], columns=methods, dtype=object)
    for c in cats:
        for m in methods:
            table.loc[c, m] = f"{mean.loc[(c, m), 'image_auroc']:.3f} ± {std.loc[(c, m), 'image_auroc']:.3f}"
    for m in methods:
        v = per_seed[m]
        table.loc["mean", m] = f"{v.mean():.3f} ± {(v.std() if len(v) > 1 else 0.0):.3f}"
    table.to_csv(os.path.join(out_dir, "summary_image_auroc.csv"))

    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(cats) * len(methods) / 2), 4))
    x = np.arange(len(cats))
    wbar = 0.8 / len(methods)
    for i, m in enumerate(methods):
        mu = [mean.loc[(c, m), "image_auroc"] for c in cats]
        sd = [std.loc[(c, m), "image_auroc"] for c in cats]
        ax.bar(x + (i - (len(methods) - 1) / 2) * wbar, mu, wbar, yerr=sd, capsize=3, label=m)
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(cats); ax.set_ylim(0, 1.05)
    ax.set_ylabel("image AUROC"); ax.set_title(f"Image AUROC (mean ± std over {df['seed'].nunique()} seeds)")
    ax.legend(fontsize=8, ncol=len(methods), loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "summary_image_auroc.png"), dpi=300); plt.close(fig)

    pt_cols = [c for c in df.columns if c.startswith("img_auroc_")]
    long = df.melt(id_vars=["category", "strategy", "seed"], value_vars=pt_cols,
                   var_name="defect_type", value_name="auroc").dropna()
    long["defect_type"] = long["defect_type"].str[len("img_auroc_"):]
    pt = long.groupby(["category", "defect_type", "strategy"])["auroc"].agg(["mean", "std"]).fillna(0.0)
    pt.round(4).to_csv(os.path.join(out_dir, "per_defect_type_summary.csv"))

    log(f"\nImage AUROC (mean ± std over seeds):\n{table.to_string()}")
    log(f"\nImage AUROC per real defect type (mean over seeds):\n"
        f"{pt['mean'].unstack('strategy')[methods].round(3).to_string()}")


# =====================================================================================
# Main
# =====================================================================================
def prepare(args):
    log("Downloading ResNet18 ImageNet weights...")
    resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    log(f"Downloading DTD to {args.dtd_root} ...")
    from torchvision.datasets import DTD
    DTD(args.dtd_root, split="train", download=True)
    log("Done. Compute nodes can now run without internet.")


def main():
    args = parse_args()
    if args.prepare:
        prepare(args)
        return
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"{datetime.now():%Y-%m-%d %H:%M} | torch {torch.__version__} | device: "
        f"{torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU (no GPU found!)'}")
    log("config:", vars(args))

    strategies = [s for s in args.strategies if s in STRATEGIES]
    dtd = load_dtd(args, args.img_size) if "perlin" in strategies else None
    if strategies:
        save_poster_grid(args, strategies, dtd)

    results_csv = os.path.join(args.out_dir, "results.csv")
    all_rows = []

    for cat in args.categories:
        cat_out = os.path.join(args.out_dir, cat)
        os.makedirs(cat_out, exist_ok=True)
        data = load_category(args.data_root, cat, args.img_size)
        textures = dtd if dtd is not None else data["train"]
        gens = make_generators(textures, on_object=cat in OBJECTS)

        # if strategies:
        #     save_synthetic_samples(data, gens, strategies, cat_out, cat, args)
        save_real_defects(data, cat_out, cat, args.sample_size)
        if args.samples_only:
            continue
        hflip = cat not in NO_FLIP

        for seed in args.seeds:
            run_dir = os.path.join(cat_out, f"seed{seed}")
            os.makedirs(run_dir, exist_ok=True)
            # hold out normal images for the validation loss (the split changes with the seed)
            perm = np.random.default_rng(seed).permutation(len(data["train"]))
            n_val = int(round(args.val_frac * len(perm)))
            val_imgs, train_imgs = data["train"][perm[:n_val]], data["train"][perm[n_val:]]

            maps, hists, rows = {}, {}, []
            for s in args.strategies:
                log(f"=== {cat} / {s} / seed {seed} | {len(train_imgs)} train, {len(val_imgs)} val | hflip={hflip} ===")
                trained = s in STRATEGIES
                if trained:
                    model, hist, secs = train(s, gens[s], train_imgs, val_imgs, args, seed, hflip, run_dir, cat, device)
                    maps[s] = predict_maps(model, data["test"], device)
                    hists[s] = hist
                    if args.save_all_ckpts or seed == args.seeds[0]:
                        torch.save(model.state_dict(), f"{run_dir}/{s}.pt")
                    del model
                else:
                    maps[s], secs = knn_baseline(train_imgs, data["test"], args.img_size, device)
                metrics = evaluate(maps[s], data)
                last_val = hists[s]["val_loss"][-1] if trained else None
                row = {"time": datetime.now().strftime("%Y-%m-%d %H:%M"), "category": cat, "strategy": s,
                       "type": "trained" if trained else "training-free", "seed": seed,
                       "epochs": args.epochs if trained else 0, "img_size": args.img_size,
                       "lr": args.lr if trained else NAN, "enc_lr_mult": args.enc_lr_mult if trained else NAN,
                       "batch": args.batch_size if trained else NAN, "anomaly_p": args.anomaly_p if trained else NAN,
                       "dtd": dtd is not None if s == "perlin" else NAN,
                       "final_train_loss": hists[s]["train_loss"][-1] if trained else NAN,
                       "final_val_loss": last_val if last_val is not None else NAN,
                       "minutes": round(secs / 60, 2), **metrics}
                rows.append(row)
                all_rows.append(row)
                append_row(results_csv, row)
                log(f"  -> image AUROC {metrics['image_auroc']:.3f} | pixel AUROC {metrics['pixel_auroc']:.3f} | "
                    f"pixel AP {metrics['pixel_ap']:.3f}")
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            res = pd.DataFrame(rows).set_index("strategy")
            save_seed_plots(res, hists, maps, data, args.strategies, run_dir, cat)
            log(f"[{cat} / seed {seed}] results:\n"
                f"{res[['image_auroc', 'pixel_auroc', 'pixel_ap']].round(3).to_string()}")

    if all_rows:
        write_summary(all_rows, args.strategies, args.out_dir)
    log("Done.")


if __name__ == "__main__":
    main()
