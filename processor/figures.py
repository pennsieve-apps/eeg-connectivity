"""Figures: connectivity matrix per band, circle plot of the strongest edges, node-strength topomaps,
and hemispheric asymmetry across recordings."""
from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np

from processor import channels as C
from processor import connectivity as K

RESEARCH = "Research output, to be reviewed by a clinician."


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _grid(n: int, max_cols: int = 3):
    cols = min(n, max_cols)
    rows = int(math.ceil(n / cols))
    return rows, cols


def matrices(path: Path, rid: str, names: list[str], contacts, mats: dict[str, np.ndarray], measure: str,
             bipolar: bool) -> None:
    """One heatmap per band of |connectivity|, channels ordered left (front to back) | midline | right."""
    plt = _plt()
    order = C.display_order(names, contacts)
    ix = [names.index(n) for n in order]
    sides = [C.hemisphere(contacts.get(n, ())) for n in order]
    cuts = [i for i in range(1, len(order)) if sides[i] != sides[i - 1]]
    rows, cols = _grid(len(mats))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.3 * rows), squeeze=False)
    fs = 6 if len(order) > 20 else 7
    for ax, (band, m) in zip(axes.ravel(), mats.items()):
        w = K.weights(m)[np.ix_(ix, ix)]
        d = w.copy()
        np.fill_diagonal(d, np.nan)
        vmax = float(np.nanmax(d)) if np.isfinite(d).any() and np.nanmax(d) > 0 else 1.0
        im = ax.imshow(d, cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest")
        for c in cuts:
            ax.axhline(c - 0.5, color="white", lw=0.8)
            ax.axvline(c - 0.5, color="white", lw=0.8)
        ax.set_xticks(range(len(order)))
        ax.set_yticks(range(len(order)))
        ax.set_xticklabels(order, rotation=90, fontsize=fs)
        ax.set_yticklabels(order, fontsize=fs)
        ax.set_title(band, fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.75)
    for ax in axes.ravel()[len(mats):]:
        ax.set_axis_off()
    note = " (bipolar derivations)" if bipolar else ""
    fig.suptitle(f"{rid}{note}\n{K.MEASURE_LABEL[measure]} per band; white lines separate left | midline | right",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def circles(path: Path, rid: str, names: list[str], contacts, mats: dict[str, np.ndarray], measure: str,
            n_lines: int, bipolar: bool) -> None:
    """Circle plot of the n_lines strongest edges per band: left hemisphere on the left half (front at
    the top), right hemisphere mirrored on the right half, midline at the bottom."""
    import mne
    from mne_connectivity.viz import plot_connectivity_circle

    plt = _plt()
    order = C.display_order(names, contacts)
    sides = [C.hemisphere(contacts.get(n, ())) for n in order]
    # counter-clockwise from the top: left front to back, midline, right back to front (mirrored)
    order = [n for n, s in zip(order, sides) if s != "R"] + [n for n, s in zip(order, sides) if s == "R"][::-1]
    sides = [C.hemisphere(contacts.get(n, ())) for n in order]
    n_left = sides.count("L")
    n_mid = sum(1 for s in sides if s not in ("L", "R"))
    bounds = [0, n_left, n_left + n_mid] if n_mid else [0, n_left]
    bounds = sorted(set(b for b in bounds if b < len(order)))
    angles = mne.viz.circular_layout(order, order, start_pos=90 + 180 / max(1, len(order)),
                                     group_boundaries=bounds if len(bounds) > 1 else None)
    angles = {n: a for n, a in zip(order, angles)}
    node_angles = np.array([angles[n] for n in names])
    colors = [{"L": "#4C78A8", "R": "#E45756"}.get(C.hemisphere(contacts.get(n, ())), "#9D9D9D") for n in names]
    rows, cols = _grid(len(mats))
    fig = plt.figure(figsize=(4.4 * cols, 4.6 * rows), facecolor="white")
    for k, (band, m) in enumerate(mats.items()):
        ax = fig.add_subplot(rows, cols, k + 1, projection="polar")
        w = K.weights(m)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plot_connectivity_circle(w, names, n_lines=n_lines, node_angles=node_angles, node_colors=colors,
                                     facecolor="white", textcolor="black", node_edgecolor="white",
                                     colormap="viridis", vmin=0, vmax=float(w.max()) if w.max() > 0 else 1.0,
                                     colorbar=True, colorbar_size=0.3, colorbar_pos=(-0.25, 0.05),
                                     fontsize_names=6, fontsize_colorbar=6, padding=4.0, ax=ax,
                                     interactive=False, show=False, linewidth=1.8)
        ax.set_title(band, fontsize=9, color="black", pad=14)
    note = " (bipolar derivations)" if bipolar else ""
    fig.suptitle(f"{rid}{note}\n{n_lines} strongest edges per band ({K.MEASURE_LABEL[measure]}); "
                 "blue = left, red = right, grey = midline", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def _info(names: list[str], contacts: dict[str, tuple[str, ...]]):
    import mne

    pos = {}
    for n in names:
        p = C.position(contacts.get(n, ()))
        # one marker per place: a reversed bipolar pair shares its midpoint
        if p is not None and not any(np.allclose(p, q, atol=1e-4) for q in pos.values()):
            pos[n] = p
    if len(pos) < 4:
        return None, []
    tpl = C.montage().get_positions()
    dig = mne.channels.make_dig_montage(ch_pos=pos, nasion=tpl["nasion"], lpa=tpl["lpa"], rpa=tpl["rpa"],
                                        coord_frame="head")
    keep = list(pos)
    info = mne.create_info(keep, 256.0, "eeg")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info.set_montage(dig, on_missing="ignore")
    return info, keep


def strength_topomaps(path: Path, rid: str, names: list[str], contacts, strengths: dict[str, np.ndarray],
                      measure: str, bipolar: bool) -> bool:
    """One scalp map of node strength per band. False when positions are missing."""
    import mne

    info, keep = _info(names, contacts)
    if info is None:
        return False
    plt = _plt()
    rows, cols = _grid(len(strengths), 5)
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 3.5 * rows), squeeze=False)
    idx = [names.index(k) for k in keep]
    for ax, (band, values) in zip(axes.ravel(), strengths.items()):
        v = np.asarray(values, float)[idx]
        ok = np.isfinite(v)
        if ok.sum() < 3:
            ax.set_axis_off()
            ax.set_title(f"{band}\n(not enough values)", fontsize=9)
            continue
        v = np.where(ok, v, np.nanmedian(v))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            im, _ = mne.viz.plot_topomap(v, info, axes=ax, show=False, cmap="viridis", names=keep, contours=0,
                                         sensors=True, vlim=(float(np.nanmin(v)), float(np.nanmax(v))))
        for t in ax.texts:
            t.set_fontsize(5)
        ax.set_title(band, fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.7)
    for ax in axes.ravel()[len(strengths):]:
        ax.set_axis_off()
    note = "; bipolar pairs at their midpoints" if bipolar else ""
    fig.suptitle(f"{rid}\nnode strength: mean {K.MEASURE_LABEL[measure]} with every other channel{note}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def asymmetry_overview(path: Path, rows: list[tuple[str, str, float]], bands: list[str], measure: str) -> bool:
    """Heatmap of the hemispheric asymmetry (L - R)/(L + R) per recording and band."""
    rids = sorted({r for r, _, _ in rows})
    if not rids:
        return False
    plt = _plt()
    m = np.full((len(rids), len(bands)), np.nan)
    for r, b, v in rows:
        if b in bands:
            m[rids.index(r), bands.index(b)] = v
    lim = float(np.nanmax(np.abs(m))) if np.isfinite(m).any() else 0.1
    lim = max(lim, 0.02)
    fig, ax = plt.subplots(figsize=(1.2 * len(bands) + 3.8, 0.32 * len(rids) + 1.6))
    im = ax.imshow(m, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels(bands, fontsize=8)
    ax.set_yticks(range(len(rids)))
    ax.set_yticklabels(rids, fontsize=6)
    for i in range(len(rids)):
        for j in range(len(bands)):
            if np.isfinite(m[i, j]):
                ax.text(j, i, f"{m[i, j]:+.2f}", ha="center", va="center", fontsize=5.5,
                        color="white" if abs(m[i, j]) > 0.6 * lim else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="(left - right) / (left + right)")
    ax.set_title(f"hemispheric asymmetry of within-hemisphere {K.MEASURE_LABEL[measure]}\n"
                 "red: more connectivity within the left hemisphere; blue: within the right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True
