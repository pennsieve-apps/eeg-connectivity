"""
Exploratory check on the Siena demo data (not a validation of a lateralising use): does the sign
of the hemispheric connectivity asymmetry agree with the side of the annotated seizures?

For each patient, the side of the seizures is read from the `channels` column of the seizure
rows in eeg-prep's `_events.tsv` (odd electrode numbers = left, even = right; a seizure whose
channels span both sides is bilateral; a patient's side is the majority of its one-sided
seizures). The asymmetry (hemisphere.csv, (left - right) / (left + right)) is averaged over the
patient's runs per measure and band, and counted as agreeing when it is positive for a left-sided
patient or negative for a right-sided one.

Usage: python validation/siena_side.py <eeg-prep eeg folder> <eeg-connectivity output folder>
  e.g. python validation/siena_side.py ../workflow-scalp-eeg/data/work/siena/eeg-prep/eeg \
           ../workflow-scalp-eeg/data/work/siena/eeg-connectivity/eeg-connectivity
Needs pandas.
"""
from __future__ import annotations

import collections
import csv
import re
import sys
from pathlib import Path

import pandas as pd


def seizure_sides(prep: Path) -> dict[str, str]:
    sides: dict[str, list[str]] = {}
    for ev in sorted(prep.rglob("*_events.tsv")):
        sub = re.match(r"(sub-[^_]+)_", ev.name).group(1)
        with open(ev, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                if r.get("trial_type") != "seizure":
                    continue
                chans = [c.strip() for c in (r.get("channels") or "").split(",") if c.strip() not in ("", "n/a")]
                s = {("L" if int(m.group(1)) % 2 else "R") for c in chans for m in [re.search(r"(\d+)$", c)] if m}
                sides.setdefault(sub, []).append(s.pop() if len(s) == 1 else "B")
    out = {}
    for sub, ss in sides.items():
        c = collections.Counter(ss)
        out[sub] = "L" if c["L"] > c["R"] else "R" if c["R"] > c["L"] else "B"
    return out


def main(prep: Path, out: Path) -> None:
    side = seizure_sides(prep)
    print("seizure side per patient:", dict(sorted(side.items())))
    h = pd.read_csv(out / "hemisphere.csv")
    h["sub"] = h.recording_id.str.extract(r"^(sub-[^_]+)")
    m = h.groupby(["sub", "measure", "band"]).asymmetry.mean().reset_index()
    m["side"] = m["sub"].map(side)
    lat = m[m.side.isin(["L", "R"])].copy()
    lat["agree"] = ((lat.asymmetry > 0) & (lat.side == "L")) | ((lat.asymmetry < 0) & (lat.side == "R"))
    print(lat.groupby(["measure", "band"]).agg(patients=("agree", "size"), agree=("agree", "sum"))
          .reset_index().to_string(index=False))
    print(lat.pivot_table(index=["sub", "side"], columns=["measure", "band"], values="asymmetry").round(3).to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
