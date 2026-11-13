"""
eeg-connectivity: sensor-space functional connectivity of clinical scalp EEG.

  INPUT_DIR   the eeg-prep BIDS-EEG derivative (recordings.csv + EDF + sidecars), any BIDS-EEG
              dataset, or loose EDF/BDF/BrainVision/FIF/EEGLAB files; found recursively
  OUTPUT_DIR/eeg-connectivity[/LABEL]/
      connectivity.csv  node_strength.csv  graph.csv  hemisphere.csv  features.csv
      figures/*.png  summary.json  provenance.json

One recording at a time, read a chunk at a time (twice: once to find artifact-free epochs, once
to stream them into the estimators); a recording that fails is logged and skipped.
Exit codes: 0 ok, 1 no usable input / every recording failed, 2 bad parameters.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _prepare_environment() -> None:
    """Lambda-safe start: HOME and TMPDIR may be fresh, empty folders and everything else
    read-only. Writable caches go under TMPDIR; nothing is written to the image."""
    tmp = Path(os.environ.get("TMPDIR") or "/tmp")
    for var, default in (("TMPDIR", tmp), ("HOME", tmp / "home")):
        p = Path(os.environ.get(var) or default)
        try:
            p.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        os.environ[var] = str(p)
    mpl = Path(os.environ["TMPDIR"]) / "mpl"
    baked = Path(os.environ.get("EEG_CONNECTIVITY_MPL_CACHE", "/opt/eeg-connectivity/matplotlib"))
    if not os.environ.get("MPLCONFIGDIR") or not os.access(os.environ["MPLCONFIGDIR"], os.W_OK):
        mpl.mkdir(parents=True, exist_ok=True)
        if baked.is_dir():  # the font cache built at image build time: copy instead of rebuilding
            for f in baked.glob("fontlist-*.json"):
                target = mpl / f.name
                if not target.exists():
                    target.write_bytes(f.read_bytes())
        os.environ["MPLCONFIGDIR"] = str(mpl)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
    os.environ.setdefault("MNE_USE_CUDA", "false")


_prepare_environment()

from processor import resources as R  # noqa: E402  (no numpy import)

# Threads before numpy: THREADS=0 -> from the cgroup CPU quota, capped by memory.
_requested = os.environ.get("THREADS", "0").strip() or "0"
try:
    _THREADS, _THREADS_WHY = R.resolve_workers(int(_requested), per_worker_mb=300, reserve_mb=700)
except ValueError:
    _THREADS, _THREADS_WHY = 1, f"THREADS={_requested!r} is not an integer"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = str(_THREADS)

import csv  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from importlib import metadata as importlib_metadata  # noqa: E402

import numpy as np  # noqa: E402

from processor import channels as C  # noqa: E402
from processor import connectivity as K  # noqa: E402
from processor import figures as F  # noqa: E402
from processor import inputs as I  # noqa: E402
from processor import windows as W  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
log = logging.getLogger("eeg-connectivity")
log.setLevel(logging.INFO)
for _noisy in ("mne", "matplotlib", "mne_connectivity"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

APP_NAME = "eeg-connectivity"
REPO_ROOT = Path(__file__).resolve().parent.parent
TRUE, FALSE = ("1", "true", "yes", "on"), ("0", "false", "no", "off")
MIN_SFREQ = 100.0
PAD_S = 8.0          # envelope filters: continuous EEG read on each side of a chunk, within the interval

CITATIONS = ["10.1016/j.neuroimage.2011.01.055", "10.1016/j.clinph.2004.04.029", "10.1038/nn.3101",
             "10.1016/j.neuroimage.2009.10.003", "10.1016/j.nicl.2017.09.021", "10.5281/zenodo.10278399",
             "10.3389/fnins.2013.00267"]

BIPOLAR_WARNING = (
    "bipolar-only recording: connectivity is computed between bipolar derivations, not electrodes. "
    "Neighbouring derivations share an electrode (Fp1-F7 and F7-T7 share F7), so their signals are not "
    "independent; wPLI, imaginary coherence and orthogonalised envelope correlation ignore the zero-lag "
    "part the shared electrode adds, but values, strengths and graph measures are not comparable with "
    "referential (average-reference) recordings")

DEFAULTS = {
    "MEASURES": "wpli", "BANDS": "conventions", "EPOCH_S": "4", "DENSITY": "0.2", "N_NULL": "20", "SEED": "42",
    "TOP_EDGES": "20", "EXCLUDE_SEIZURE": "true", "SEIZURE_MARGIN_S": "60", "REJECT": "true", "REJECT_UV": "500",
    "MAX_EPOCHS": "0", "EXCLUDE_CHANNELS": "A1,A2,M1,M2", "FIGURES": "true", "CHUNK_S": "60", "THREADS": "0", "LABEL": "",
}

HEADERS = {
    "connectivity": ["recording_id", "channel_a", "channel_b", "band", "measure", "value"],
    "node_strength": ["recording_id", "channel", "band", "measure", "strength"],
    "graph": ["recording_id", "band", "measure", "metric", "value"],
    "hemisphere": ["recording_id", "band", "measure", "left", "right", "interhemispheric", "asymmetry"],
    "features": ["recording_id", "channel", "feature", "value"],
}


def app_version() -> str:
    try:
        return (REPO_ROOT / "VERSION").read_text().strip()
    except OSError:
        return "unknown"


def tool_versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("mne", "mne-connectivity", "networkx", "numpy", "scipy", "matplotlib"):
        try:
            out[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            pass
    return out


def _bool(name: str, v: str) -> bool:
    s = str(v).strip().lower()
    if s in TRUE:
        return True
    if s in FALSE:
        return False
    raise ValueError(f"{name} must be true or false, got {v!r}")


def read_params(env=os.environ) -> tuple[dict, W.WindowParams, dict]:
    raw = {k: (env.get(k) if env.get(k) not in (None, "") else v) for k, v in DEFAULTS.items()}
    try:
        wp = W.WindowParams(epoch_s=float(raw["EPOCH_S"]), reject=_bool("REJECT", raw["REJECT"]),
                            reject_uv=float(raw["REJECT_UV"]), chunk_s=float(raw["CHUNK_S"]))
        other = {"measures": K.parse_measures(raw["MEASURES"]), "bands": W.parse_bands(raw["BANDS"]),
                 "density": float(raw["DENSITY"]), "n_null": int(raw["N_NULL"]), "seed": int(raw["SEED"]),
                 "top_edges": int(raw["TOP_EDGES"]), "max_epochs": int(raw["MAX_EPOCHS"]),
                 "exclude_seizure": _bool("EXCLUDE_SEIZURE", raw["EXCLUDE_SEIZURE"]),
                 "margin_s": float(raw["SEIZURE_MARGIN_S"]), "figures": _bool("FIGURES", raw["FIGURES"]),
                 "label": raw["LABEL"].strip(),
                 "exclude_channels": [] if raw["EXCLUDE_CHANNELS"].strip().lower() in ("none", "-") else
                 [c.strip() for c in raw["EXCLUDE_CHANNELS"].split(",") if c.strip()]}
        int(raw["THREADS"])
    except (TypeError, ValueError) as e:
        raise ValueError(str(e)) from None
    if not 0 < other["density"] <= 1:
        raise ValueError("DENSITY must be in (0, 1]")
    if other["n_null"] < 0 or other["top_edges"] < 1 or other["max_epochs"] < 0:
        raise ValueError("N_NULL and MAX_EPOCHS must be >= 0, TOP_EDGES >= 1")
    if other["margin_s"] < 0:
        raise ValueError("SEIZURE_MARGIN_S must be >= 0")
    for c in other["exclude_channels"]:
        if C.electrode(c) is None:
            raise ValueError(f"EXCLUDE_CHANNELS: {c!r} is not a 10-20/10-10 electrode name")
    other["exclude_channels"] = [C.electrode(c) for c in other["exclude_channels"]]
    if other["label"] and not all(c.isalnum() or c in "-_" for c in other["label"]):
        raise ValueError("LABEL: letters, digits, - and _ only")
    if min(b[1] for b in other["bands"]) < 2.0 / wp.epoch_s:
        raise ValueError(f"the lowest band edge must be at least 2 cycles per epoch ({2 / wp.epoch_s:g} Hz "
                         f"for EPOCH_S={wp.epoch_s:g})")
    return raw, wp, other


def fmt(x, nd: int = 6) -> str:
    if x is None:
        return ""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "" if not math.isfinite(xf) else f"{xf:.{nd}g}"


def num(x, nd: int = 4):
    """A JSON number (None when not finite)."""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return round(xf, nd) if math.isfinite(xf) else None


# --------------------------------------------------------------------------- channels and montage
# (as eeg-spectral)

def select_channels(raw_names: list[str], rec: I.Recording, exclude: list[str] = ()):
    """(raw indices, canonical names, contacts by name, montage, reference, notes). Channels with a
    contact in `exclude` (EXCLUDE_CHANNELS: ear and mastoid electrodes by default) are left out."""
    montage, mapping, reference = C.classify(raw_names)
    notes = []
    dropped = [n for n in raw_names if n not in mapping]
    if dropped:
        notes.append(f"{len(dropped)} non-EEG or unrecognised channel(s) left out: {', '.join(dropped[:8])}"
                     + (" ..." if len(dropped) > 8 else ""))
    excluded = set()
    if rec.channels is not None:
        by = {c.name.strip().lower(): c for c in rec.channels}
        for raw_name in mapping:
            c = by.get(raw_name.strip().lower())
            if c is not None and (c.status == "bad" or c.type not in ("EEG", "")):
                excluded.add(raw_name)
        if excluded:
            notes.append(f"{len(excluded)} channel(s) marked bad in _channels.tsv: {', '.join(sorted(excluded))}")
    off = [n for n in mapping if n not in excluded and set(mapping[n][1]) & set(exclude)]
    if off:
        notes.append(f"{len(off)} channel(s) on ear/mastoid or other EXCLUDE_CHANNELS electrodes left out: "
                     f"{', '.join(mapping[n][0] for n in off)}")
        excluded |= set(off)
    idx, names, contacts = [], [], {}
    for i, n in enumerate(raw_names):
        if n in mapping and n not in excluded:
            canon, cont = mapping[n]
            idx.append(i)
            names.append(canon)
            contacts[canon] = cont
    if rec.montage:
        label = rec.montage
    elif montage == "bipolar-longitudinal":
        label = "bipolar-longitudinal"
    elif reference:
        label = "as-recorded"
    elif any(n.lower().endswith(("-avg", "_avg", " avg")) for n in raw_names):
        label = "average"
    else:
        label = "as-recorded"
    if not reference:
        reference = rec.reference or ("average" if label == "average" else "")
    return idx, names, contacts, label, reference, notes


def unit_scale(raw, idx: list[int], notes: list[str]) -> float:
    """1e-6 when an EDF without a physical dimension holds microvolts (MNE then assumes volts:
    ds003555), else 1. Decided from a 10 s stretch in the middle (as eeg-spectral)."""
    orig = getattr(raw, "_orig_units", None) or {}
    names = [raw.ch_names[i] for i in idx]
    if not names or any(str(orig.get(n, "n/a")).lower() not in ("n/a", "", "none") for n in names):
        return 1.0
    sf = raw.info["sfreq"]
    mid = raw.n_times // 2
    a, b = max(0, mid - int(5 * sf)), min(raw.n_times, mid + int(5 * sf))
    x = raw.get_data(picks=idx, start=a, stop=b, units="uV", verbose="error")
    if np.median(np.std(x, axis=1)) > 1e4:
        notes.append("EDF without a physical dimension: values read as µV (they would be implausible as V)")
        return 1e-6
    return 1.0


def sidecar_reference(rec: I.Recording) -> str:
    js = I.sidecar(rec.path, "eeg.json")
    if not js:
        return ""
    try:
        ref = str(json.loads(js.read_text()).get("EEGReference", ""))
    except (OSError, ValueError):
        return ""
    return "average" if "average" in ref.lower() else ""


# --------------------------------------------------------------------------- one recording

def process(rec: I.Recording, wp: W.WindowParams, other: dict, out: Path, writers: dict) -> dict:
    import mne

    t0 = time.monotonic()
    raw = mne.io.read_raw(str(rec.path), preload=False, verbose="error")
    sfreq = float(raw.info["sfreq"])
    duration = raw.n_times / sfreq
    rid = rec.recording_id
    if rec.source == "loose":
        I.load_annotations(rec, raw)
    if sfreq < MIN_SFREQ:
        raise ValueError(f"sampling rate {sfreq:g} Hz is below {MIN_SFREQ:g} Hz")
    idx, names, contacts, montage, reference, notes = select_channels(raw.ch_names, rec, other["exclude_channels"])
    if montage == "as-recorded" and not rec.montage and sidecar_reference(rec) == "average":
        montage, reference = "average", "average"
    if len(idx) < 4:
        raise ValueError(f"{len(idx)} usable EEG channel(s) (10-20/10-10 names) of {len(raw.ch_names)}; need 4")
    bipolar = montage == "bipolar-longitudinal"
    if bipolar:
        notes.append(BIPOLAR_WARNING)
        log.warning("%s: %s", rid, BIPOLAR_WARNING)
    elif montage != "average":
        notes.append(f"montage {montage!r} (reference {reference or 'unknown'}): re-referenced to the average of "
                     "the channels used, as connectivity is computed in average reference")
    bands = [b for b in other["bands"] if b[2] < sfreq / 2 * 0.9]
    if len(bands) < len(other["bands"]):
        notes.append(f"band(s) above 90 % of Nyquist left out: "
                     f"{', '.join(b[0] for b in other['bands'] if b not in bands)}")
    if not bands:
        raise ValueError("no band below the Nyquist frequency")
    scale = unit_scale(raw, idx, notes)

    def read(a: int, b: int, sel: np.ndarray | None = None) -> np.ndarray:
        picks = idx if sel is None else [idx[i] for i in sel]
        x = raw.get_data(picks=picks, start=a, stop=b, units="uV", verbose="error")
        if scale != 1.0:
            x = x * scale
        if not bipolar:
            x = x - x.mean(axis=0, keepdims=True)       # average reference over the channels used
        return x

    intervals = W.analysable(duration, rec.bad, rec.seizures, other["exclude_seizure"], other["margin_s"])
    last_log = [time.monotonic()]

    def progress(done: int, total: int, what: str = "epochs screened"):
        if time.monotonic() - last_log[0] > 30:
            log.info("%s: %d of %d %s", rid, done, total, what)
            last_log[0] = time.monotonic()

    # pass 1: epochs and rejection. The average reference here is over every selected channel; a
    # channel the rejection then drops (flat or noisy in most epochs) is left out of the average in
    # pass 2, so its noise can affect which epochs pass 1 rejects but not the estimates.
    ep = W.compute_epochs(read, sfreq, len(idx), intervals, wp, progress)
    try:
        rinfo = W.reject(ep, wp)
    finally:
        ep.close()
    ok = np.flatnonzero(ep.channel_ok)
    if rinfo.get("rejected_channels"):
        bad = [names[i] for i in rinfo["rejected_channels"]]
        notes.append(f"{len(bad)} channel(s) flat or above {wp.reject_uv:g} µV in most epochs, left out: {', '.join(bad)}")
    keep = ep.keep.copy()
    n_screened = len(keep)
    if other["max_epochs"] and keep.sum() > other["max_epochs"]:
        kept = np.flatnonzero(keep)
        pick = kept[np.round(np.linspace(0, len(kept) - 1, other["max_epochs"])).astype(int)]
        keep[:] = False
        keep[pick] = True
        notes.append(f"{len(kept)} artifact-free epochs; MAX_EPOCHS={other['max_epochs']} spread evenly over the "
                     "recording were used")
    if keep.sum() < K.MIN_EPOCHS:
        raise ValueError(f"{int(keep.sum())} artifact-free {wp.epoch_s:g} s epoch(s) of {n_screened}; "
                         f"need at least {K.MIN_EPOCHS}")
    if len(ok) < 4:
        raise ValueError(f"{len(ok)} usable channel(s) after rejection; need 4")
    ch_names = [names[i] for i in ok]
    n_ch = len(ch_names)

    # pass 2: stream the kept epochs into the estimators
    measures = other["measures"]
    aec = K.EnvelopeAccumulator(sfreq, bands, n_ch) if "aec" in measures else None
    nper = ep.nper
    ivs = [(int(math.ceil(a * sfreq)), int(math.floor(b * sfreq))) for a, b in intervals]
    per_chunk = max(1, int(wp.chunk_s * sfreq) // nper)
    pad = int(PAD_S * sfreq)
    n_used = int(keep.sum())

    def epochs():
        k0, done = 0, 0
        for (ia, ib), s in zip(ivs, ep.starts):
            for c0 in range(0, len(s), per_chunk):
                block = s[c0: c0 + per_chunk]
                kb = keep[k0 + c0: k0 + c0 + len(block)]
                if not kb.any():
                    continue
                kept = block[kb]
                a = max(ia, int(kept[0]) - pad) if aec else int(kept[0])
                b = min(ib, int(kept[-1]) + nper + pad) if aec else int(kept[-1]) + nper
                x = read(a, b, ok)
                if aec is not None:
                    aec.add(x, (kept - a).tolist(), nper)
                for st in kept:
                    o = int(st) - a
                    yield x[:, o: o + nper]
                done += len(kept)
                progress(done, n_used, "epochs analysed")
            k0 += len(s)

    mats: dict[str, np.ndarray] = {}
    if any(m in K.SPECTRAL for m in measures):
        mats.update(K.spectral_measures(epochs(), sfreq, n_ch, bands, measures))
    else:
        K.iterate(epochs())
    if aec is not None:
        mats["aec"] = aec.result()
    mats = {m: mats[m] for m in measures}
    band_names = [b[0] for b in bands]

    # tables
    pairs = C.homologous_pairs(ch_names, contacts)
    col = {ch: i for i, ch in enumerate(ch_names)}
    hemi_left, hemi_right = [col[a] for a, _ in pairs], [col[b] for _, b in pairs]
    paired = {a for a, _ in pairs} | {b for _, b in pairs}
    unpaired = [ch for ch in ch_names if C.hemisphere(contacts[ch]) in ("L", "R") and ch not in paired]
    if unpaired:
        notes.append(f"hemisphere means use {len(pairs)} homologous pair(s); left out because their mirror channel "
                     f"is missing: {', '.join(unpaired)}")
    iu = np.triu_indices(n_ch, 1)
    headline: dict = {"montage": montage, "n_channels": n_ch, "seconds_analysed": num(n_used * wp.epoch_s, 1),
                      "epochs_used": n_used, "epochs_rejected": int(rinfo.get("rejected_epochs", 0))}
    strengths: dict[str, dict[str, np.ndarray]] = {}
    for m, arr in mats.items():
        strengths[m] = {}
        for b, band in enumerate(band_names):
            v = arr[:, :, b]
            for i, j in zip(*iu):
                writers["connectivity"].writerow([rid, ch_names[i], ch_names[j], band, m, fmt(v[i, j])])
            st = K.node_strength(v)
            strengths[m][band] = st
            for c, ch in enumerate(ch_names):
                writers["node_strength"].writerow([rid, ch, band, m, fmt(st[c])])
                writers["features"].writerow([rid, ch, f"{m}_{band}", fmt(st[c])])
            g = K.graph_metrics(v, other["density"], other["n_null"], other["seed"])
            for metric in K.GRAPH_METRICS:
                writers["graph"].writerow([rid, band, m, metric, fmt(g[metric])])
            h = K.hemisphere_summary(v, hemi_left, hemi_right)
            writers["hemisphere"].writerow([rid, band, m, fmt(h["left"]), fmt(h["right"]),
                                            fmt(h["interhemispheric"]), fmt(h["asymmetry"])])
            w = K.weights(v)[iu]
            headline[f"{m}_{band}_mean"] = num(w.mean(), 4)
            headline[f"{m}_{band}_asymmetry"] = num(h["asymmetry"], 4)
            headline[f"{m}_{band}_small_worldness"] = num(g["small_worldness"], 3)
    primary = measures[0]
    if "alpha" in band_names:
        top = K.strongest_edges(mats[primary][:, :, band_names.index("alpha")], ch_names, 1)
        if top:
            headline[f"{primary}_alpha_strongest_edge"] = f"{top[0][0]}–{top[0][1]}"

    # figures (primary measure)
    figs = []
    if other["figures"]:
        try:
            figs = make_figures(out, rid, ch_names, contacts, mats[primary], band_names, strengths[primary],
                                primary, other["top_edges"], bipolar, other["density"])
        except Exception as e:  # noqa: BLE001 - a figure problem must not lose the tables
            log.warning("%s: figures failed: %s", rid, e)
            notes.append(f"figures failed: {e}")

    secs = time.monotonic() - t0
    log.info("%s: %d EEG channels (%s) of %d, %g Hz, %d of %d epochs used (%.0f s of %.0f s), %s alpha "
             "asymmetry %s, %.1fs", rid, n_ch, montage, len(raw.ch_names), sfreq, n_used, n_screened,
             n_used * wp.epoch_s, duration, primary, fmt(headline.get(f"{primary}_alpha_asymmetry"), 3), secs)
    return {"status": "ok", "seconds": round(secs, 2), "montage": montage, "reference": reference,
            "sfreq": sfreq, "duration_s": round(duration, 3), "n_channels_file": len(raw.ch_names),
            "channels": ch_names, "bands": {b[0]: [b[1], b[2]] for b in bands},
            "analysable_intervals_s": [[round(a, 3), round(b, 3)] for a, b in intervals],
            "seizures_excluded": len(rec.seizures) if other["exclude_seizure"] else 0,
            "bad_segments": len(rec.bad), **rinfo,
            "hemisphere_pairs": [f"{a}/{b}" for a, b in pairs], "hemisphere_unpaired": unpaired, "epochs_used": n_used,
            "seconds_analysed": round(n_used * wp.epoch_s, 2),
            "notes": notes + rec.notes, "_headline": headline, "_figures": figs,
            "_asym": [(rid, band, headline[f"{primary}_{band}_asymmetry"]) for band in band_names]}


def make_figures(out: Path, rid, ch_names, contacts, arr, band_names, strengths, measure, top_edges, bipolar,
                 density) -> list[dict]:
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    figs = []
    label = K.MEASURE_LABEL[measure]
    mats = {band: arr[:, :, b] for b, band in enumerate(band_names)}
    bip = ("Channels are bipolar derivations; neighbouring derivations share an electrode, so compare only with "
           "other bipolar recordings. " if bipolar else "")
    name = f"{rid}_matrix.png"
    F.matrices(figdir / name, rid, ch_names, contacts, mats, measure, bipolar)
    figs.append({"path": f"figures/{name}", "caption": (
        f"{rid}: functional connectivity ({label}) between every pair of channels, one panel per frequency band. "
        "Channels run from the left hemisphere (front to back) through the midline to the right hemisphere, so "
        "the top-left block is connectivity within the left hemisphere, the bottom-right block within the right, "
        "and the off-diagonal blocks between hemispheres. Brighter = more consistent phase lag between the two "
        "channels; zero-lag coupling from volume conduction is not counted. The figure describes the recording's "
        "network; a difference between the left and right blocks is not a lateralising finding (on the demo data the "
        f"side of the asymmetry did not match the side of the seizures better than chance). {bip}{F.RESEARCH}")})
    name = f"{rid}_edges.png"
    F.circles(figdir / name, rid, ch_names, contacts, mats, measure, top_edges, bipolar)
    figs.append({"path": f"figures/{name}", "caption": (
        f"{rid}: the {top_edges} strongest connections ({label}) per band, drawn on a circle with the left "
        "hemisphere on the left (frontal at the top), the right hemisphere mirrored on the right and midline "
        "channels at the bottom. Lines crossing the circle are interhemispheric; lines staying on one side are "
        "within a hemisphere. The plot shows where the strongest connections of this recording are; a cluster on "
        f"one side is a description of the sensor network, not a lateralising finding. {bip}{F.RESEARCH}")})
    name = f"{rid}_strength.png"
    if F.strength_topomaps(figdir / name, rid, ch_names, contacts, strengths, measure, bipolar):
        figs.append({"path": f"figures/{name}", "caption": (
            f"{rid}: node strength per band, the mean {label} of each channel with every other channel, on a scalp "
            "map. A hot spot means that channel is more synchronised with the rest of the scalp than its neighbours. "
            "Left-right differences in node strength are not a lateralising finding. Values are relative "
            "within the recording (colour scale per band). "
            + ("Bipolar pairs are drawn at the midpoint of their two electrodes. " if bipolar else "") + F.RESEARCH)})
    return figs


# --------------------------------------------------------------------------- main

def peak_rss_mb() -> float:
    for f in ("/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"):
        try:
            return round(int(Path(f).read_text().strip()) / 2**20, 1)
        except (OSError, ValueError):
            continue
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / 2**20 if sys.platform == "darwin" else r / 1024, 1)


def main() -> int:
    t_start = time.monotonic()
    started = datetime.now(timezone.utc)
    input_dir = Path(os.environ.get("INPUT_DIR", "/data/input"))
    output_dir = Path(os.environ.get("OUTPUT_DIR", "/data/output"))
    try:
        raw_params, wp, other = read_params()
    except ValueError as e:
        log.error("parameters: %s", e)
        return 2
    out = output_dir / APP_NAME / other["label"] if other["label"] else output_dir / APP_NAME
    out.mkdir(parents=True, exist_ok=True)
    log.info("%s %s: INPUT_DIR=%s OUTPUT_DIR=%s measures=%s threads=%d (%s)", APP_NAME, app_version(), input_dir,
             out, ",".join(other["measures"]), _THREADS, _THREADS_WHY)
    recs = I.find_recordings(input_dir) if input_dir.is_dir() else []
    if not recs:
        log.error("no recording found under %s (recordings.csv, *_eeg.edf/.bdf/.vhdr/.fif/.set or loose files)",
                  input_dir)
        return 1
    log.info("%d recording(s) found", len(recs))

    def rel(pth: Path) -> str:
        try:
            return str(pth.relative_to(input_dir))
        except ValueError:
            return str(pth)

    per_rec: dict[str, dict] = {}
    headlines: dict[str, dict] = {}
    figures: list[dict] = []
    asym_rows: list = []
    warnings_out: list[str] = []
    n_ok = 0
    handles = {k: open(out / f"{k}.csv", "w", newline="", encoding="utf-8") for k in HEADERS}
    try:
        writers = {k: csv.writer(h, lineterminator="\n") for k, h in handles.items()}
        for k, w in writers.items():
            w.writerow(HEADERS[k])
        for i, rec in enumerate(recs, 1):
            log.info("[%d/%d] %s (%s)", i, len(recs), rec.recording_id, rel(rec.path))
            try:
                res = process(rec, wp, other, out, writers)
                headlines[rec.recording_id] = res.pop("_headline")
                figures.extend(res.pop("_figures"))
                asym_rows.extend(res.pop("_asym"))
                per_rec[rec.recording_id] = {"path": rel(rec.path), **res}
                for n in res["notes"]:
                    warnings_out.append(f"{rec.recording_id}: {n}")
                n_ok += 1
            except Exception as e:  # noqa: BLE001 - a recording that fails is logged and skipped
                log.error("%s: failed: %s", rec.recording_id, e)
                log.debug(traceback.format_exc())
                per_rec[rec.recording_id] = {"path": rel(rec.path), "status": "failed", "error": str(e)}
                warnings_out.append(f"{rec.recording_id}: failed: {e}")
            for h in handles.values():
                h.flush()
    finally:
        for h in handles.values():
            h.close()

    primary = other["measures"][0]
    band_names = [b[0] for b in other["bands"]]
    if other["figures"] and len({r for r, _, _ in asym_rows}) > 1:
        try:
            (out / "figures").mkdir(parents=True, exist_ok=True)
            if F.asymmetry_overview(out / "figures" / "asymmetry_overview.png",
                                    [(r, b, v if v is not None else math.nan) for r, b, v in asym_rows],
                                    band_names, primary):
                figures.insert(0, {"path": "figures/asymmetry_overview.png", "caption": (
                    f"Hemispheric asymmetry of connectivity ({K.MEASURE_LABEL[primary]}) for every recording "
                    "(rows) and band (columns): (left - right) / (left + right) of the mean connectivity within "
                    "each hemisphere, over homologous channel pairs present on both sides (midline left out). Red: "
                    "more connectivity within the left hemisphere; blue: within the right. This describes the "
                    "sensor-level network; it is not a lateralising finding. It is a sensor-level proxy of the "
                    "source-level network asymmetry studied in temporal lobe epilepsy (Verhoeven et al. 2018), and "
                    "on the Siena demo data its sign matched the side of the annotated seizures only at chance "
                    f"level. {F.RESEARCH}")})
        except Exception as e:  # noqa: BLE001
            log.warning("asymmetry overview figure failed: %s", e)

    finished = datetime.now(timezone.utc)
    provenance = {
        "app": APP_NAME, "version": app_version(), "variant": None, "tools": tool_versions(),
        "parameters": {**raw_params, "connectivity": {
            "measures": list(other["measures"]), "bands": {n: [lo, hi] for n, lo, hi in other["bands"]},
            "epoch_s": wp.epoch_s, "spectral": "mne-connectivity spectral_connectivity_epochs, multitaper "
            "(default bandwidth), faverage", "envelope": "4th-order Butterworth band-pass (zero phase), Hilbert, "
            "mne-connectivity envelope_correlation(orthogonalize='pairwise', log=True), weighted mean over chunks",
            "graph": {"threshold": "proportional", "density": other["density"], "n_null": other["n_null"],
                      "null": "degree-preserving double edge swaps, weights shuffled", "seed": other["seed"]},
            "reject": wp.reject, "reject_uv": wp.reject_uv, "flat_uv": W.FLAT_UV, "robust_z": W.ROBUST_Z,
            "robust_z_band_hz": list(W.MUSCLE_BAND)}},
        "inputs": sorted(rel(r.path) for r in recs),
        "started": started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "finished": finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "wall_seconds": round(time.monotonic() - t_start, 2), "peak_rss_mb": peak_rss_mb(),
        "threads": {"requested": int(raw_params["THREADS"]), "used": _THREADS, "why": _THREADS_WHY},
        "recordings": per_rec, "citations": CITATIONS, "licence_notice": None,
    }
    with open(out / "provenance.json", "w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=2, ensure_ascii=False)
    tables = [
        {"path": "connectivity.csv", "caption": "Connectivity of every channel pair per band and measure (wpli 0-1; "
                                                "imcoh signed, -1 to 1, negative when channel_a leads channel_b; "
                                                "aec 0-1)."},
        {"path": "node_strength.csv", "caption": "Node strength: mean |connectivity| of each channel with every other "
                                                 "channel, per band and measure."},
        {"path": "graph.csv", "caption": f"Graph measures of the proportionally thresholded network (density "
                                         f"{other['density']:g}): weighted clustering, characteristic path length, "
                                         "global efficiency, Louvain modularity, small-worldness against "
                                         "degree-preserving null networks, density."},
        {"path": "hemisphere.csv", "caption": "Mean |connectivity| within the left hemisphere, within the right "
                                              "(over homologous channel pairs present on both sides), "
                                              "between them, and the asymmetry (left - right)/(left + right)."},
        {"path": "features.csv", "caption": "Node strength per channel in the shared long shape (feature "
                                            "<measure>_<band>)."},
    ]
    summary = {"app": APP_NAME, "version": app_version(), "recordings": headlines, "figures": figures,
               "tables": tables, "warnings": warnings_out,
               "notice": "Connectivity and network measures are research outputs, to be reviewed by a clinician; "
                         "they are not a clinical EEG interpretation and have not been validated for "
                         "lateralisation or diagnosis."}
    with open(out / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    log.info("%d of %d recording(s) done in %.1fs, peak memory %.0f MB; wrote %s", n_ok, len(recs),
             time.monotonic() - t_start, provenance["peak_rss_mb"], out)
    if n_ok == 0:
        log.error("every recording failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
