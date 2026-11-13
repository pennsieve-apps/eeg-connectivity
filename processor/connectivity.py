"""
Sensor-space functional connectivity of one recording, its graph measures and its hemispheric
summary.

Measures (MEASURES):
- `wpli`: weighted phase lag index (Vinck et al. 2011), and `imcoh`: imaginary part of coherency
  (Nolte et al. 2004; in connectivity.csv a negative value means channel_a leads channel_b), both
  from mne-connectivity's `spectral_connectivity_epochs` (multitaper, default bandwidth: 7 DPSS
  tapers on 4 s epochs, ±1 Hz smoothing), averaged over the bins of each band (`faverage`). Both
  ignore the zero-lag (real) part of the cross-spectrum, which is what volume conduction and a
  shared reference produce.
- `aec`: orthogonalised amplitude envelope correlation (Hipp et al. 2012), from
  mne-connectivity's `envelope_correlation(orthogonalize="pairwise", log=True)` (correlation of the
  log power envelopes, as Hipp et al. did) on the analytic signal of the band-passed EEG (4th-order Butterworth, zero phase, applied to each chunk of continuous
  analysable EEG, then Hilbert). Correlations are taken over the kept epochs of each chunk
  (CHUNK_S) and averaged over chunks, weighted by their length.

Weights for strength and graphs are |value| (imcoh is signed: its sign is the direction of the
phase lag and says nothing about strength).

Graph measures (Rubinov & Sporns 2010 definitions, computed with networkx): the matrix is
thresholded proportionally (the DENSITY strongest edges are kept), weights are normalised to the
strongest edge; clustering = mean weighted clustering coefficient (Onnela's geometric-mean
definition, as networkx's `average_clustering(weight=)`); path_length = characteristic path
length with edge length 1/weight (mean over connected pairs); global_efficiency = mean of
1/shortest weighted path over all pairs (0 when disconnected); modularity = Newman's Q of the
Louvain partition (seeded); small_worldness = (C / C_null) / (L / L_null) against N_NULL
degree-preserving rewired graphs (double edge swaps, the original weights shuffled over the
rewired edges); density = edges kept / possible edges.
"""
from __future__ import annotations

import math
import warnings
from typing import Iterable, Iterator

import numpy as np

MEASURES = ("wpli", "imcoh", "aec")
SPECTRAL = {"wpli": "wpli", "imcoh": "imcoh"}      # our name -> mne-connectivity method
MEASURE_LABEL = {"wpli": "wPLI", "imcoh": "|imaginary coherence|", "aec": "orthogonalised envelope correlation"}
MIN_EPOCHS = 10


def parse_measures(value: str) -> tuple[str, ...]:
    v = (value or "wpli").strip().lower()
    if v == "all":
        return MEASURES
    out = []
    for m in (x.strip() for x in v.replace(";", ",").split(",")):
        if not m:
            continue
        if m in ("envelope", "envelope_correlation", "orth-aec", "oaec"):
            m = "aec"
        if m not in MEASURES:
            raise ValueError(f"MEASURES: {m!r} is not one of {', '.join(MEASURES)} (or all)")
        if m not in out:
            out.append(m)
    if not out:
        raise ValueError("MEASURES is empty")
    return tuple(out)


# --------------------------------------------------------------------------- spectral measures

def spectral_measures(epochs: Iterable[np.ndarray], sfreq: float, n_ch: int, bands, measures) -> dict[str, np.ndarray]:
    """{measure: (n_ch, n_ch, n_bands) symmetric, NaN diagonal} for wpli/imcoh, streaming the epochs
    (each (n_ch, n_times) µV) through mne-connectivity one at a time."""
    from mne_connectivity import spectral_connectivity_epochs

    methods = [SPECTRAL[m] for m in measures if m in SPECTRAL]
    fmin = tuple(b[1] for b in bands)
    fmax = tuple(b[2] for b in bands)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = spectral_connectivity_epochs(epochs, method=methods, sfreq=sfreq, mode="multitaper", fmin=fmin,
                                           fmax=fmax, faverage=True, n_jobs=1, verbose="error")
    if not isinstance(res, list):
        res = [res]
    out = {}
    names = [m for m in measures if m in SPECTRAL]
    for name, con in zip(names, res):
        d = np.asarray(con.get_data(output="dense"), float)   # (n, n, n_bands), lower triangle
        d = d + np.transpose(d, (1, 0, 2))
        idx = np.arange(n_ch)
        d[idx, idx, :] = np.nan
        out[name] = d
    return out


class EnvelopeAccumulator:
    """Orthogonalised envelope correlation per band, accumulated chunk by chunk."""

    def __init__(self, sfreq: float, bands, n_ch: int):
        from scipy.signal import butter

        self.sfreq = sfreq
        self.bands = bands
        self.sos = [butter(4, [lo, hi], btype="bandpass", fs=sfreq, output="sos") for _, lo, hi in bands]
        self.sum = np.zeros((n_ch, n_ch, len(bands)))
        self.weight = 0.0

    def add(self, x: np.ndarray, starts: Iterable[int], nper: int) -> None:
        """x: (n_ch, n) continuous µV; starts: offsets in x of the kept epochs."""
        from mne_connectivity import envelope_correlation
        from scipy.signal import hilbert, sosfiltfilt

        starts = list(starts)
        if not starts:
            return
        sel = (np.asarray(starts)[:, None] + np.arange(nper)[None, :]).ravel()
        for b, sos in enumerate(self.sos):
            z = hilbert(sosfiltfilt(sos, x, axis=-1), axis=-1)[:, sel]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c = envelope_correlation(z[np.newaxis], orthogonalize="pairwise", log=True, verbose="error")
            self.sum[:, :, b] += np.asarray(c.get_data(output="dense"), float).reshape(x.shape[0], x.shape[0]) * len(sel)
        self.weight += len(sel)

    def result(self) -> np.ndarray:
        if self.weight == 0:
            return np.full_like(self.sum, np.nan)
        out = self.sum / self.weight
        out = (out + np.transpose(out, (1, 0, 2))) / 2
        idx = np.arange(out.shape[0])
        out[idx, idx, :] = np.nan
        return out


# --------------------------------------------------------------------------- node and hemisphere

def weights(values: np.ndarray) -> np.ndarray:
    """|value| with a zero diagonal (n, n)."""
    w = np.abs(np.asarray(values, float))
    w = np.where(np.isfinite(w), w, 0.0)
    np.fill_diagonal(w, 0.0)
    return w


def node_strength(values: np.ndarray) -> np.ndarray:
    """Mean |connectivity| of each channel with every other channel (strength / (N - 1))."""
    w = weights(values)
    n = w.shape[0]
    return w.sum(axis=1) / max(1, n - 1)


def hemisphere_summary(values: np.ndarray, left: list[int], right: list[int]) -> dict[str, float]:
    """Mean |connectivity| within the left channels, within the right channels, and between left
    and right; asymmetry = (left - right) / (left + right). `left` and `right` are matched
    homologous channels (left[k] mirrors right[k], channels.homologous_pairs), so both sides are
    averaged over the same set of electrode positions; midline and unpaired channels are left out."""
    w = weights(values)
    L, R = np.asarray(left, int), np.asarray(right, int)

    def within(ix):
        if len(ix) < 2:
            return math.nan
        sub = w[np.ix_(ix, ix)]
        return float(sub[np.triu_indices(len(ix), 1)].mean())

    lv, rv = within(L), within(R)
    inter = float(w[np.ix_(L, R)].mean()) if len(L) and len(R) else math.nan
    tot = lv + rv
    asym = (lv - rv) / tot if math.isfinite(tot) and tot > 0 else math.nan
    return {"left": lv, "right": rv, "interhemispheric": inter, "asymmetry": asym}


# --------------------------------------------------------------------------- graphs

def proportional_threshold(values: np.ndarray, density: float) -> np.ndarray:
    """Keep the round(density * possible) strongest edges (ties broken by order), normalised to the
    strongest edge; zero elsewhere."""
    w = weights(values)
    n = w.shape[0]
    iu = np.triu_indices(n, 1)
    e = w[iu]
    k = max(1, int(round(density * len(e))))
    order = np.argsort(-e, kind="stable")[:k]
    order = order[e[order] > 0]
    a = np.zeros_like(w)
    a[iu[0][order], iu[1][order]] = e[order]
    a = a + a.T
    mx = a.max() if a.size else 0.0
    return a / mx if mx > 0 else a


def _graph(a: np.ndarray):
    import networkx as nx

    g = nx.Graph()
    n = a.shape[0]
    g.add_nodes_from(range(n))
    iu = np.triu_indices(n, 1)
    for i, j in zip(*iu):
        if a[i, j] > 0:
            g.add_edge(int(i), int(j), weight=float(a[i, j]), length=1.0 / float(a[i, j]))
    return g


def _clustering(g) -> float:
    import networkx as nx

    return float(nx.average_clustering(g, weight="weight")) if g.number_of_nodes() else math.nan


def _paths(g) -> tuple[float, float]:
    """(characteristic path length over connected pairs, global efficiency over all pairs)."""
    import networkx as nx

    n = g.number_of_nodes()
    if n < 2:
        return math.nan, math.nan
    dist = dict(nx.all_pairs_dijkstra_path_length(g, weight="length"))
    ds = [d for i, row in dist.items() for j, d in row.items() if i != j]
    L = float(np.mean(ds)) if ds else math.nan
    E = float(np.sum([1.0 / d for d in ds if d > 0]) / (n * (n - 1)))
    return L, E


def graph_metrics(values: np.ndarray, density: float, n_null: int, seed: int) -> dict[str, float]:
    import networkx as nx

    a = proportional_threshold(values, density)
    g = _graph(a)
    n = g.number_of_nodes()
    m = g.number_of_edges()
    out = {"clustering": math.nan, "path_length": math.nan, "global_efficiency": math.nan,
           "modularity": math.nan, "small_worldness": math.nan,
           "density": m / (n * (n - 1) / 2) if n > 1 else math.nan}
    if m == 0:
        return out
    C = _clustering(g)
    L, E = _paths(g)
    out.update(clustering=C, path_length=L, global_efficiency=E)
    try:
        comms = nx.community.louvain_communities(g, weight="weight", seed=seed)
        out["modularity"] = float(nx.community.modularity(g, comms, weight="weight"))
    except Exception:  # noqa: BLE001
        pass
    if n_null > 0 and n >= 4 and m >= 2:
        rng = np.random.default_rng(seed)
        ws = np.array([d["weight"] for _, _, d in g.edges(data=True)])
        cs, ls = [], []
        for _ in range(n_null):
            h = nx.Graph()
            h.add_nodes_from(g.nodes())
            h.add_edges_from(g.edges())
            try:
                nx.double_edge_swap(h, nswap=10 * m, max_tries=1000 * m, seed=int(rng.integers(2**31)))
            except (nx.NetworkXError, nx.NetworkXAlgorithmError):
                continue
            for (i, j), w in zip(h.edges(), rng.permutation(ws)):
                h[i][j]["weight"] = float(w)
                h[i][j]["length"] = 1.0 / float(w)
            cs.append(_clustering(h))
            ls.append(_paths(h)[0])
        if cs:
            cr, lr = float(np.nanmean(cs)), float(np.nanmean(ls))
            if cr > 0 and lr > 0 and math.isfinite(L) and L > 0:
                out["small_worldness"] = (C / cr) / (L / lr)
    return out


GRAPH_METRICS = ("clustering", "path_length", "global_efficiency", "modularity", "small_worldness", "density")


def strongest_edges(values: np.ndarray, names: list[str], k: int) -> list[tuple[str, str, float]]:
    w = weights(values)
    n = w.shape[0]
    iu = np.triu_indices(n, 1)
    e = w[iu]
    order = np.argsort(-e, kind="stable")[:k]
    return [(names[iu[0][o]], names[iu[1][o]], float(np.asarray(values)[iu[0][o], iu[1][o]])) for o in order]


def iterate(gen: Iterator) -> None:
    for _ in gen:
        pass
