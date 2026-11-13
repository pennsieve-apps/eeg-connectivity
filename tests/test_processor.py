"""The synthetic fixture through the processor (no Docker): planted couplings are recovered."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest

from processor import channels as C
from processor import connectivity as K
from processor import main as M
from processor import windows as W
from tests import make_fixture


def rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="session")
def fixture(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("fixture")
    make_fixture.main(d)
    return d


def run(monkeypatch, input_dir: Path, output_dir: Path, **env) -> int:
    for k in M.DEFAULTS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("INPUT_DIR", str(input_dir))
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    return M.main()


@pytest.fixture(scope="session")
def out_all(fixture, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("out")
    mp = pytest.MonkeyPatch()
    try:
        assert run(mp, fixture / "eeg", out, MEASURES="all") == 0
    finally:
        mp.undo()
    return out / "eeg-connectivity"


@pytest.fixture(scope="session")
def truth(fixture) -> dict:
    return json.loads((fixture / "truth.json").read_text())


def by(rs, **kw):
    return [r for r in rs if all(r[k] == v for k, v in kw.items())]


def edges(out: Path, rid: str, band: str, measure: str) -> dict[frozenset, float]:
    return {frozenset((r["channel_a"], r["channel_b"])): float(r["value"])
            for r in by(rows(out / "connectivity.csv"), recording_id=rid, band=band, measure=measure)}


# ---------------------------------------------------------------- planted couplings

@pytest.mark.parametrize("measure,minimum", [("wpli", 0.55), ("imcoh", 0.4), ("aec", 0.4)])
def test_lagged_pairs_are_the_strongest_edges(out_all, truth, measure, minimum):
    rid = truth["SYN01"]["recording_id"]
    for lag in truth["lagged"]:
        e = edges(out_all, rid, lag["band"], measure)
        pair = frozenset(lag["channels"])
        best = max(e, key=lambda k: abs(e[k]))
        assert best == pair, (measure, lag, best, e[best])
        assert abs(e[pair]) > minimum, (measure, lag, e[pair])


@pytest.mark.parametrize("measure,maximum", [("wpli", 0.3), ("imcoh", 0.06), ("aec", 0.15)])
def test_zero_lag_copies_are_ignored(out_all, truth, measure, maximum):
    """Volume conduction (the same signal on several electrodes at zero lag) is not connectivity."""
    rid = truth["SYN01"]["recording_id"]
    for zl in truth["zero_lag"]:
        e = edges(out_all, rid, zl["band"], measure)
        chans = [c for c in zl["channels"] if c != truth["SYN01"]["bad_channel"]]
        for i, a in enumerate(chans):
            for b in chans[i + 1:]:
                assert abs(e[frozenset((a, b))]) < maximum, (measure, zl["band"], a, b, e[frozenset((a, b))])


def test_lagged_coupling_on_bipolar_recording(out_all, truth):
    rid = truth["SYN02"]["recording_id"]
    for measure in ("wpli", "imcoh", "aec"):
        e = edges(out_all, rid, "theta", measure)
        best = max(e, key=lambda k: abs(e[k]))
        contacts = [set(ch.split("-")) for ch in best]
        assert any("T7" in c for c in contacts) and any("O1" in c for c in contacts), (measure, best)
        assert abs(e[best]) > 0.5


def test_hemispheric_asymmetry_points_left(out_all, truth):
    """Every planted lagged coupling is within the left hemisphere; the zero-lag copies are on the right."""
    hemi = rows(out_all / "hemisphere.csv")
    for key in ("SYN01", "SYN02"):
        rid = truth[key]["recording_id"]
        for measure in ("wpli", "imcoh", "aec"):
            for band in ("theta", "beta"):
                h = by(hemi, recording_id=rid, measure=measure, band=band)[0]
                assert float(h["asymmetry"]) > 0.15, (key, measure, band, h)
                assert float(h["left"]) > float(h["right"])
            # no planted asymmetry in alpha or gamma
            for band in ("alpha", "gamma"):
                h = by(hemi, recording_id=rid, measure=measure, band=band)[0]
                assert abs(float(h["asymmetry"])) < 0.15, (key, measure, band, h)


def test_node_strength_and_features(out_all, truth):
    rid = truth["SYN01"]["recording_id"]
    ns = {r["channel"]: float(r["strength"]) for r in
          by(rows(out_all / "node_strength.csv"), recording_id=rid, band="theta", measure="wpli")}
    assert len(ns) == 18
    top2 = sorted(ns, key=ns.get, reverse=True)[:2]
    assert set(top2) == {"T7", "O1"}
    ft = rows(out_all / "features.csv")
    assert list(ft[0]) == ["recording_id", "channel", "feature", "value"]
    assert {r["feature"] for r in ft} == {f"{m}_{b}" for m in ("wpli", "imcoh", "aec")
                                          for b in ("delta", "theta", "alpha", "beta", "gamma")}
    f = {r["channel"]: float(r["value"]) for r in by(ft, recording_id=rid, feature="wpli_theta")}
    assert f == pytest.approx(ns)


def test_graph_measures(out_all, truth):
    g = rows(out_all / "graph.csv")
    rid = truth["SYN01"]["recording_id"]
    sel = by(g, recording_id=rid, band="theta", measure="wpli")
    vals = {r["metric"]: r["value"] for r in sel}
    assert set(vals) == set(K.GRAPH_METRICS)
    assert float(vals["density"]) == pytest.approx(round(0.2 * 153) / 153, abs=1e-5)
    for k in ("clustering", "path_length", "global_efficiency", "modularity", "small_worldness"):
        assert vals[k] != "" and np.isfinite(float(vals[k])), k
    assert 0 < float(vals["clustering"]) <= 1 and 0 < float(vals["global_efficiency"]) <= 1


def test_connectivity_table_shape(out_all, truth):
    c = rows(out_all / "connectivity.csv")
    assert list(c[0]) == ["recording_id", "channel_a", "channel_b", "band", "measure", "value"]
    rid = truth["SYN01"]["recording_id"]
    assert len(by(c, recording_id=rid)) == 153 * 5 * 3
    w = [float(r["value"]) for r in by(c, measure="wpli")]
    assert min(w) >= 0 and max(w) <= 1
    assert min(float(r["value"]) for r in by(c, measure="imcoh")) < 0          # imcoh is signed


# ---------------------------------------------------------------- what is left out

def test_bad_channel_bad_segment_seizure_pop(out_all, truth):
    prov = json.loads((out_all / "provenance.json").read_text())
    rec = prov["recordings"][truth["SYN01"]["recording_id"]]
    assert "Cz" not in rec["channels"] and len(rec["channels"]) == 18
    assert rec["analysable_intervals_s"] == [[0.0, 60.0], [70.0, 240.0]]
    assert rec["seizures_excluded"] == 1 and rec["bad_segments"] == 1
    assert rec["montage"] == "average"
    assert rec["epochs"] == 57 and rec["rejected_epochs"] >= 1 and rec["rejected_pop_epochs"] >= 1
    assert rec["epochs_used"] == rec["epochs"] - rec["rejected_epochs"]
    bip = prov["recordings"][truth["SYN02"]["recording_id"]]
    assert bip["montage"] == "bipolar-longitudinal" and bip["channels"][0] == "Fp1-F7"


def test_seizure_would_change_right_theta(fixture, tmp_path, monkeypatch, truth):
    """The planted seizure spreads over the right hemisphere with a lag: included, it raises right
    theta wPLI; left out (default), it does not."""
    assert run(monkeypatch, fixture / "eeg", tmp_path, EXCLUDE_SEIZURE="false", FIGURES="false") == 0
    rid = truth["SYN01"]["recording_id"]
    out = tmp_path / "eeg-connectivity"
    with_sz = by(rows(out / "hemisphere.csv"), recording_id=rid, band="theta", measure="wpli")[0]
    e = edges(out, rid, "theta", "wpli")
    assert e[frozenset(("C4", "P4"))] > 0.5
    assert float(with_sz["right"]) > 0.25


def test_bipolar_warning(out_all, truth):
    s = json.loads((out_all / "summary.json").read_text())
    rid = truth["SYN02"]["recording_id"]
    assert any(w.startswith(rid) and "bipolar-only recording" in w and "share an electrode" in w
               for w in s["warnings"])
    assert not any(w.startswith(truth["SYN01"]["recording_id"]) and "bipolar-only" in w for w in s["warnings"])


def test_summary_and_provenance(out_all, truth):
    s = json.loads((out_all / "summary.json").read_text())
    assert s["app"] == "eeg-connectivity" and s["version"] == "0.1.0"
    head = s["recordings"][truth["SYN01"]["recording_id"]]
    assert isinstance(head["wpli_theta_asymmetry"], float) and head["wpli_theta_asymmetry"] > 0
    assert head["wpli_alpha_strongest_edge"]
    assert len(s["figures"]) == 7 and s["figures"][0]["path"] == "figures/asymmetry_overview.png"
    for fig in s["figures"]:
        assert (out_all / fig["path"]).is_file() and len(fig["caption"]) > 80
        assert "research output" in fig["caption"].lower()
    for t in s["tables"]:
        assert (out_all / t["path"]).is_file()
    prov = json.loads((out_all / "provenance.json").read_text())
    assert set(prov) == {"app", "version", "variant", "tools", "parameters", "inputs", "started", "finished",
                         "wall_seconds", "peak_rss_mb", "threads", "recordings", "citations", "licence_notice"}
    assert prov["tools"]["mne-connectivity"] and prov["tools"]["networkx"] and prov["citations"]
    assert set(prov["threads"]) == {"requested", "used", "why"}


def test_default_is_wpli_only(fixture, tmp_path, monkeypatch, truth):
    assert run(monkeypatch, fixture / "eeg", tmp_path, FIGURES="false") == 0
    out = tmp_path / "eeg-connectivity"
    assert {r["measure"] for r in rows(out / "connectivity.csv")} == {"wpli"}
    assert not (out / "figures").exists()


# ---------------------------------------------------------------- other inputs

def test_plain_bids_siena_style(fixture, tmp_path, monkeypatch, truth):
    assert run(monkeypatch, fixture / "plain", tmp_path, SEIZURE_MARGIN_S="10") == 0
    out = tmp_path / "eeg-connectivity"
    prov = json.loads((out / "provenance.json").read_text())
    rec = prov["recordings"][truth["SIE01"]["recording_id"]]
    assert rec["montage"] == "average"
    assert "T7" in rec["channels"] and "P8" in rec["channels"] and "T3" not in rec["channels"]
    assert "EKG" not in " ".join(rec["channels"]) and len(rec["channels"]) == 19
    assert rec["analysable_intervals_s"] == [[0.0, 90.0], [130.0, 150.0]]
    e = edges(out, truth["SIE01"]["recording_id"], "theta", "wpli")
    assert max(e, key=e.get) == frozenset(("T7", "O1"))


def test_loose_bipolar_edf_with_chbmit_summary(fixture, tmp_path, monkeypatch, truth):
    assert run(monkeypatch, fixture / "loose", tmp_path, FIGURES="false", SEIZURE_MARGIN_S="10") == 0
    out = tmp_path / "eeg-connectivity"
    prov = json.loads((out / "provenance.json").read_text())
    rec = prov["recordings"][truth["chb99"]["recording_id"]]
    assert rec["montage"] == "bipolar-longitudinal" and rec["seizures_excluded"] == 1
    assert rec["analysable_intervals_s"] == [[0.0, 70.0], [105.0, 120.0]]


def test_too_few_epochs_fails_the_recording(fixture, tmp_path, monkeypatch, truth):
    """chb99 with the default 60 s margin keeps 20 s (5 epochs): logged as failed, exit 1."""
    assert run(monkeypatch, fixture / "loose", tmp_path, FIGURES="false") == 1
    prov = json.loads((tmp_path / "eeg-connectivity" / "provenance.json").read_text())
    r = prov["recordings"][truth["chb99"]["recording_id"]]
    assert r["status"] == "failed" and "epoch" in r["error"]


def test_everything_side_by_side(fixture, tmp_path, monkeypatch):
    """Ports merged at the root of INPUT_DIR, as symlinks."""
    inp = tmp_path / "in"
    inp.mkdir()
    for d in ("eeg", "plain", "loose"):
        os.symlink(fixture / d, inp / d)
    assert run(monkeypatch, inp, tmp_path / "out", FIGURES="false", SEIZURE_MARGIN_S="10") == 0
    prov = json.loads((tmp_path / "out" / "eeg-connectivity" / "provenance.json").read_text())
    assert len(prov["recordings"]) == 4 and all(r["status"] == "ok" for r in prov["recordings"].values())


def test_label_adds_a_level(fixture, tmp_path, monkeypatch):
    assert run(monkeypatch, fixture / "eeg", tmp_path, LABEL="second", FIGURES="false") == 0
    assert (tmp_path / "eeg-connectivity" / "second" / "features.csv").is_file()


def test_max_epochs(fixture, tmp_path, monkeypatch, truth):
    assert run(monkeypatch, fixture / "eeg", tmp_path, MAX_EPOCHS="20", FIGURES="false") == 0
    prov = json.loads((tmp_path / "eeg-connectivity" / "provenance.json").read_text())
    assert all(r["epochs_used"] == 20 for r in prov["recordings"].values())


def test_no_input_exit_1(tmp_path, monkeypatch):
    (tmp_path / "in").mkdir()
    assert run(monkeypatch, tmp_path / "in", tmp_path / "out") == 1


@pytest.mark.parametrize("env", [{"MEASURES": "coherence"}, {"BANDS": "{bad"}, {"DENSITY": "0"},
                                 {"DENSITY": "1.5"}, {"REJECT": "maybe"}, {"THREADS": "x"}, {"EPOCH_S": "0.5"},
                                 {"N_NULL": "-1"}, {"BANDS": '{"slow": [0.2, 1]}'}, {"LABEL": "a/b"},
                                 {"EXCLUDE_CHANNELS": "X9"}])
def test_bad_parameters_exit_2(fixture, tmp_path, monkeypatch, env):
    assert run(monkeypatch, fixture / "loose", tmp_path, **env) == 2


def test_failed_recording_is_skipped(fixture, tmp_path, monkeypatch):
    inp = tmp_path / "in"
    inp.mkdir()
    os.symlink(fixture / "loose" / "chb99_01.edf", inp / "chb99_01.edf")
    os.symlink(fixture / "loose" / "chb99-summary.txt", inp / "chb99-summary.txt")
    (inp / "broken_02.edf").write_bytes(b"not an edf")
    assert run(monkeypatch, inp, tmp_path / "out", FIGURES="false", SEIZURE_MARGIN_S="10") == 0
    prov = json.loads((tmp_path / "out" / "eeg-connectivity" / "provenance.json").read_text())
    assert prov["recordings"]["sub-broken_task-eeg_run-02"]["status"] == "failed"


def test_edf_without_physical_dimension_and_linked_reference(tmp_path, monkeypatch):
    """ds003555's EDFs leave the physical dimension empty (MNE then assumes volts); channels named
    'Fp1-A1' are referential to A1 and are re-referenced to the average."""
    from edfio import Edf, EdfSignal

    rng = np.random.default_rng(1)
    el = make_fixture.electrodes(120.0, rng)
    inp = tmp_path / "in"
    inp.mkdir()
    sigs = [EdfSignal(el[e], sampling_frequency=make_fixture.FS, label=f"{e}-A1",
                      physical_dimension="", physical_range=(-500, 500)) for e in make_fixture.ELECTRODES]
    Edf(sigs).write(inp / "child_01.edf")
    assert run(monkeypatch, inp, tmp_path / "out", FIGURES="false") == 0
    out = tmp_path / "out" / "eeg-connectivity"
    rec = json.loads((out / "provenance.json").read_text())["recordings"]["sub-child_task-eeg_run-01"]
    assert any("physical dimension" in n for n in rec["notes"]) and rec["rejected_epochs"] == 0
    assert any("re-referenced to the average" in n for n in rec["notes"])
    e = edges(out, "sub-child_task-eeg_run-01", "theta", "wpli")
    assert max(e, key=e.get) == frozenset(("T7", "O1"))


# ---------------------------------------------------------------- units

def test_parse_measures():
    assert K.parse_measures("wpli") == ("wpli",)
    assert K.parse_measures("all") == ("wpli", "imcoh", "aec")
    assert K.parse_measures("imcoh, envelope_correlation") == ("imcoh", "aec")
    with pytest.raises(ValueError):
        K.parse_measures("pli")


def test_proportional_threshold_and_hemisphere():
    rng = np.random.default_rng(0)
    m = rng.random((10, 10))
    m = (m + m.T) / 2
    a = K.proportional_threshold(m, 0.2)
    assert (np.triu(a, 1) > 0).sum() == 9 and a.max() == pytest.approx(1.0)
    # 2 left, 2 right, 1 midline
    v = np.array([[0, .8, .1, .1, 0], [.8, 0, .1, .1, 0], [.1, .1, 0, .2, 0], [.1, .1, .2, 0, 0], [0] * 5], float)
    h = K.hemisphere_summary(v, [0, 1], [2, 3])
    assert h["left"] == pytest.approx(0.8) and h["right"] == pytest.approx(0.2)
    assert h["interhemispheric"] == pytest.approx(0.1) and h["asymmetry"] == pytest.approx(0.6)
    assert K.node_strength(v)[0] == pytest.approx((0.8 + 0.1 + 0.1) / 4)


def test_small_world_lattice_vs_random():
    """A ring lattice with a few shortcuts is small-world (sigma > 1); the measures are finite."""
    import networkx as nx

    g = nx.connected_watts_strogatz_graph(30, 6, 0.1, seed=3)
    m = nx.to_numpy_array(g)
    res = K.graph_metrics(m, density=g.number_of_edges() / (30 * 29 / 2), n_null=10, seed=1)
    assert res["small_worldness"] > 1.5
    assert res["density"] == pytest.approx(g.number_of_edges() / 435)


def test_analysable_and_epochs():
    iv = W.analysable(360, [(60, 70)], [(300, 15)], True, 60)
    assert iv == [(0.0, 60.0), (70, 240)]
    starts = W.epoch_starts([(0, 4 * 256 * 3 + 10)], 1024)
    assert starts[0].tolist() == [0, 1024, 2048]


def test_display_order():
    names = ["Fz", "O2", "Fp1", "T7", "Fp2", "O1"]
    contacts = {n: (n,) for n in names}
    assert C.display_order(names, contacts) == ["Fp1", "T7", "O1", "Fz", "Fp2", "O2"]


def test_imcoh_sign_negative_when_channel_a_leads():
    """connectivity.csv lists each pair once (channel_a before channel_b in the recording's channel order);
    imcoh is negative when channel_a leads."""
    rng = np.random.default_rng(5)
    fs, n, lag = 256.0, 256 * 200, 8
    src = make_fixture.bandnoise(n + lag, rng, 8, 12, 10.0)
    a = src[lag:] + rng.normal(0, 3, n)          # a(t) = s(t + lag): a leads
    b = src[:n] + rng.normal(0, 3, n)
    c = rng.normal(0, 3, n)
    x = np.stack([a, b, c])
    ep = (x[:, i:i + 1024] for i in range(0, n - 1024 + 1, 1024))
    m = K.spectral_measures(ep, fs, 3, [("alpha", 8.0, 13.0)], ("imcoh",))["imcoh"][:, :, 0]
    assert m[0, 1] < -0.3                         # row 0 = channel_a (leader), column 1 = channel_b
    ep = (x[[1, 0, 2], i:i + 1024] for i in range(0, n - 1024 + 1, 1024))
    m = K.spectral_measures(ep, fs, 3, [("alpha", 8.0, 13.0)], ("imcoh",))["imcoh"][:, :, 0]
    assert m[0, 1] > 0.3                          # follower listed first: positive


def test_hemisphere_uses_homologous_pairs_only(fixture, tmp_path, monkeypatch, truth):
    """A channel bad on one side (T7) takes its mirror (T8) out of the hemisphere means."""
    import shutil

    inp = tmp_path / "eeg"
    shutil.copytree(fixture / "eeg", inp)
    for ch in inp.rglob("*SYN01*_channels.tsv"):
        lines = ch.read_text().splitlines()
        lines = [l.replace("\tgood\t", "\tbad\t") if l.startswith("T7\t") else l for l in lines]
        ch.write_text("\n".join(lines) + "\n")
    assert run(monkeypatch, inp, tmp_path / "out", FIGURES="false") == 0
    rec = json.loads((tmp_path / "out" / "eeg-connectivity" / "provenance.json").read_text())["recordings"][
        truth["SYN01"]["recording_id"]]
    assert rec["hemisphere_unpaired"] == ["T8"] and "T7/T8" not in rec["hemisphere_pairs"]
    assert len(rec["hemisphere_pairs"]) == 7 and any("mirror channel is missing" in n for n in rec["notes"])


def test_ear_electrodes_left_out(tmp_path, monkeypatch):
    from edfio import Edf, EdfSignal

    rng = np.random.default_rng(2)
    el = make_fixture.electrodes(120.0, rng)
    el["A1"], el["A2"] = rng.normal(0, 15, len(el["Fp1"])), rng.normal(0, 15, len(el["Fp1"]))
    inp = tmp_path / "in"
    inp.mkdir()
    Edf([EdfSignal(v, sampling_frequency=make_fixture.FS, label=k, physical_dimension="uV",
                   physical_range=(-500, 500)) for k, v in el.items()]).write(inp / "ears_01.edf")
    assert run(monkeypatch, inp, tmp_path / "out", FIGURES="false") == 0
    rec = json.loads((tmp_path / "out" / "eeg-connectivity" / "provenance.json").read_text())["recordings"][
        "sub-ears_task-eeg_run-01"]
    assert "A1" not in rec["channels"] and "A2" not in rec["channels"] and len(rec["channels"]) == 19
    assert any("ear/mastoid" in n for n in rec["notes"])
    assert run(monkeypatch, inp, tmp_path / "out2", FIGURES="false", EXCLUDE_CHANNELS="none") == 0
    rec = json.loads((tmp_path / "out2" / "eeg-connectivity" / "provenance.json").read_text())["recordings"][
        "sub-ears_task-eeg_run-01"]
    assert "A1" in rec["channels"]


def test_homologous_pairs_accept_reversed_bipolar_mirror():
    names = ["T7-FT9", "FT10-T8", "Fp1-F7", "Fp2-F8", "Fz-Cz"]
    contacts = {n: tuple(n.split("-")) for n in names}
    assert C.homologous_pairs(names, contacts) == [("T7-FT9", "FT10-T8"), ("Fp1-F7", "Fp2-F8")]
