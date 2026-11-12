"""
Analysable intervals, epochs and automatic artifact rejection, copied from eeg-spectral
(`processor/spectral.py`) so that both apps leave out the same EEG:

1. The analysable EEG is the recording minus `bad_segment`/`artifact` events and minus each
   annotated seizure with SEIZURE_MARGIN_S before and after (EXCLUDE_SEIZURE).
2. Whole EPOCH_S epochs (no overlap here: each epoch is one trial of the connectivity estimate)
   inside those intervals, read CHUNK_S at a time. Per epoch the peak-to-peak amplitude and the
   Welch spectrum up to 45 Hz are kept for the rejection below (memory-mapped in $TMPDIR when
   large).
3. Automatic rejection (REJECT=true), as in eeg-spectral: channels flat or above REJECT_UV in
   most epochs are left out; epochs where several channels exceed REJECT_UV, every channel is
   flat, or whose median 20-45 Hz power (muscle) is a robust outlier (z > 4) are left out. A
   connectivity estimate needs every channel in the same epoch, so an epoch where a single
   channel exceeds REJECT_UV (an electrode pop) is left out as a whole here (eeg-spectral drops
   it for that channel only).
"""
from __future__ import annotations

import json
import math
import re
import tempfile
import warnings
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

BAND_PRESETS: dict[str, tuple[tuple[str, float, float], ...]] = {
    # conventions.md (scalp): delta 1-4, theta 4-8, alpha 8-13, beta 13-30, gamma 30-45 Hz
    "conventions": (("delta", 1.0, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 13.0),
                    ("beta", 13.0, 30.0), ("gamma", 30.0, 45.0)),
}
BAND_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
FLAT_UV = 0.5                     # peak-to-peak below this in an epoch = flat
ROBUST_Z = 4.0                    # median 20-45 Hz log power above median + 4 * 1.4826 MAD
MUSCLE_BAND = (20.0, 45.0)
REJECT_FMAX = 45.0                # spectra kept for the rejection go up to here
STORE_IN_MEMORY_MB = 64           # per-epoch spectra above this go to a memmap in $TMPDIR


def parse_bands(value: str | None) -> tuple[tuple[str, float, float], ...]:
    """A preset name (conventions) or a JSON object {"delta": [1, 4], ...} (as eeg-spectral)."""
    v = (value or "conventions").strip()
    if v in BAND_PRESETS:
        return BAND_PRESETS[v]
    try:
        obj = json.loads(v)
    except json.JSONDecodeError as e:
        raise ValueError(f"BANDS must be one of {', '.join(BAND_PRESETS)} or a JSON object: {e}") from None
    if not isinstance(obj, dict) or not obj:
        raise ValueError("BANDS JSON must be a non-empty object of name: [low_hz, high_hz]")
    out = []
    for name, edges in obj.items():
        if not BAND_NAME_RE.match(name):
            raise ValueError(f"band name {name!r}: lowercase letters, digits and -")
        if not (isinstance(edges, (list, tuple)) and len(edges) == 2):
            raise ValueError(f"band {name}: [low_hz, high_hz]")
        lo, hi = float(edges[0]), float(edges[1])
        if not (0 < lo < hi):
            raise ValueError(f"band {name}: need 0 < low < high, got {lo}, {hi}")
        out.append((name, lo, hi))
    return tuple(sorted(out, key=lambda b: (b[1], b[2])))


@dataclass(frozen=True)
class WindowParams:
    epoch_s: float = 4.0
    reject: bool = True
    reject_uv: float = 500.0
    chunk_s: float = 60.0

    def __post_init__(self):
        if self.epoch_s < 1:
            raise ValueError("EPOCH_S must be at least 1 s")
        if self.reject_uv <= 0:
            raise ValueError("REJECT_UV must be > 0")
        if self.chunk_s < self.epoch_s:
            raise ValueError("CHUNK_S must be at least EPOCH_S")


def subtract(intervals: list[tuple[float, float]], cut: tuple[float, float]) -> list[tuple[float, float]]:
    out = []
    c0, c1 = cut
    for a, b in intervals:
        if c1 <= a or c0 >= b:
            out.append((a, b))
            continue
        if c0 > a:
            out.append((a, c0))
        if c1 < b:
            out.append((c1, b))
    return out


def analysable(duration: float, bad: Sequence[tuple[float, float]], seizures: Sequence[tuple[float, float | None]],
               exclude_seizure: bool, margin: float) -> list[tuple[float, float]]:
    """The recording minus bad segments/artifacts and (optionally) seizures with a margin.
    A seizure without a known end is cut to the end of the recording."""
    iv = [(0.0, duration)]
    for a, b in bad:
        iv = subtract(iv, (a, b))
    if exclude_seizure:
        for on, dur in seizures:
            iv = subtract(iv, (on - margin, (on + dur + margin) if dur else duration))
    return [(a, b) for a, b in iv if b > a]


def epoch_starts(intervals_samples: Sequence[tuple[int, int]], nper: int) -> list[np.ndarray]:
    """Per interval, the start samples of the whole, non-overlapping epochs inside it."""
    out = []
    for a, b in intervals_samples:
        if b - a < nper:
            out.append(np.zeros(0, dtype=np.int64))
        else:
            out.append(a + np.arange(0, b - a - nper + 1, nper, dtype=np.int64))
    return out


@dataclass
class Epochs:
    """Epoch bookkeeping of one recording: start samples, per-epoch spectra (n_ep, n_ch, n_f) and
    peak-to-peak amplitude (n_ep, n_ch) for the rejection, and the result of the rejection."""
    sfreq: float
    nper: int
    starts: list[np.ndarray]          # per analysable interval
    freqs: np.ndarray
    psd: np.ndarray
    ptp: np.ndarray
    keep: np.ndarray = None  # type: ignore[assignment]        (n_ep,) epoch used
    channel_ok: np.ndarray = None  # type: ignore[assignment]  (n_ch,)
    _tmp: object = None

    @property
    def all_starts(self) -> np.ndarray:
        return np.concatenate(self.starts) if self.starts else np.zeros(0, dtype=np.int64)

    def close(self):
        if self._tmp is not None:
            self.psd = None  # type: ignore[assignment]
            self._tmp.cleanup()  # type: ignore[attr-defined]
            self._tmp = None


def compute_epochs(read: Callable[[int, int], np.ndarray], sfreq: float, n_ch: int,
                   intervals_s: Sequence[tuple[float, float]], params: WindowParams,
                   progress: Callable[[int, int], None] | None = None) -> Epochs:
    """Peak-to-peak amplitude and spectrum (Hamming, one segment per epoch) of every whole epoch
    in the intervals, CHUNK_S at a time. read(start, stop) returns (n_ch, stop - start) µV."""
    from mne.time_frequency import psd_array_welch

    nper = int(round(params.epoch_s * sfreq))
    ivs = [(int(math.ceil(a * sfreq)), int(math.floor(b * sfreq))) for a, b in intervals_s]
    starts = epoch_starts(ivs, nper)
    n_ep = int(sum(len(s) for s in starts))
    freqs_all = np.fft.rfftfreq(nper, 1 / sfreq)
    freqs = freqs_all[freqs_all <= min(REJECT_FMAX, sfreq / 2) + 1e-9]
    nbytes = n_ep * n_ch * len(freqs) * 4
    tmp = None
    if nbytes > STORE_IN_MEMORY_MB * 2**20:
        tmp = tempfile.TemporaryDirectory(prefix="eeg-connectivity-")
        psd = np.lib.format.open_memmap(f"{tmp.name}/psd.npy", mode="w+", dtype=np.float32,
                                        shape=(n_ep, n_ch, len(freqs)))
    else:
        psd = np.zeros((n_ep, n_ch, len(freqs)), dtype=np.float32)
    ptp = np.zeros((n_ep, n_ch), dtype=np.float32)
    per_chunk = max(1, int(params.chunk_s * sfreq) // nper)
    w = 0
    for s in starts:
        for c0 in range(0, len(s), per_chunk):
            block = s[c0: c0 + per_chunk]
            a, b = int(block[0]), int(block[-1]) + nper
            x = np.asarray(read(a, b), dtype=np.float64)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p, _ = psd_array_welch(x, sfreq, fmin=0.0, fmax=float(freqs[-1]), n_fft=nper, n_per_seg=nper,
                                       n_overlap=0, window="hamming", average=None, verbose="error")
            psd[w: w + len(block)] = np.moveaxis(p[:, : len(freqs), : len(block)], -1, 0)
            seg = x[:, (block - a)[:, None] + np.arange(nper)[None, :]]      # (n_ch, m, nper)
            ptp[w: w + len(block)] = np.ptp(seg, axis=-1).T
            w += len(block)
            if progress:
                progress(w, n_ep)
    return Epochs(sfreq, nper, starts, freqs, psd, ptp, _tmp=tmp)


def reject(ep: Epochs, params: WindowParams) -> dict:
    """Sets ep.channel_ok and ep.keep; returns counts for provenance (eeg-spectral's rejection,
    with single-channel pops dropping the whole epoch)."""
    n_ep, n_ch = ep.ptp.shape
    info = {"epochs": int(n_ep)}
    if n_ep == 0:
        ep.channel_ok = np.ones(n_ch, bool)
        ep.keep = np.zeros(0, bool)
        info.update(rejected_epochs=0, rejected_channels=[])
        return info
    if not params.reject:
        ep.channel_ok = np.ones(n_ch, bool)
        ep.keep = np.ones(n_ep, bool)
        info.update(rejected_epochs=0, rejected_channels=[])
        return info
    high = ep.ptp > params.reject_uv
    flat = ep.ptp < FLAT_UV
    ch_bad = (high.mean(axis=0) > 0.5) | (flat.mean(axis=0) > 0.5)
    ok = ~ch_bad
    ep.channel_ok = ok
    keep = np.ones(n_ep, bool)
    if ok.sum() >= 2:
        keep &= ~high[:, ok].any(axis=1)
        keep &= ~flat[:, ok].any(axis=1)
        hi = min(MUSCLE_BAND[1], float(ep.freqs[-1]))
        bb = (ep.freqs >= min(MUSCLE_BAND[0], hi - 5)) & (ep.freqs <= hi)
        lp = np.zeros(n_ep)
        for i in range(0, n_ep, 2048):
            x = np.asarray(ep.psd[i: i + 2048][:, ok][:, :, bb], dtype=np.float64).mean(axis=2)
            lp[i: i + 2048] = np.log10(np.maximum(np.median(x, axis=1), 1e-30))
        med = np.median(lp[keep]) if keep.any() else np.median(lp)
        mad = 1.4826 * np.median(np.abs(lp - med)) if keep.any() else 0.0
        if mad > 0:
            keep &= (lp - med) / mad <= ROBUST_Z
    else:
        keep[:] = False
    ep.keep = keep
    info.update(rejected_epochs=int((~keep).sum()), rejected_channels=[int(i) for i in np.flatnonzero(ch_bad)],
                rejected_pop_epochs=int((high[:, ok].sum(axis=1) == 1).sum()) if ok.any() else 0)
    return info


def covered_seconds(n_epochs: int, epoch_s: float) -> float:
    return float(n_epochs * epoch_s)
