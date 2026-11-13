# eeg-connectivity: functional connectivity of clinical scalp EEG

eeg-connectivity is built on **MNE-Connectivity**, developed by Thomas S. Binns, Adam Li, Eric
Larson and the MNE-Connectivity contributors of the MNE tools project
([repository](https://github.com/mne-tools/mne-connectivity),
[software DOI](https://doi.org/10.5281/zenodo.10278399)), on **MNE-Python**, developed by
Alexandre Gramfort and the MNE community ([repository](https://github.com/mne-tools/mne-python),
[paper](https://doi.org/10.3389/fnins.2013.00267)), and on **NetworkX**, developed by Aric
Hagberg, Dan Schult, Pieter Swart and the NetworkX developers
([repository](https://github.com/networkx/networkx), [paper](https://doi.org/10.25080/TCWV9851)).
The connectivity measures are those of **Martin Vinck, Robert Oostenveld and colleagues** (the
weighted phase lag index; [paper](https://doi.org/10.1016/j.neuroimage.2011.01.055)), **Guido
Nolte, Mark Hallett and colleagues** (the imaginary part of coherency;
[paper](https://doi.org/10.1016/j.clinph.2004.04.029)) and **Joerg Hipp, Andreas Engel and
colleagues** (orthogonalised amplitude envelope correlation;
[paper](https://doi.org/10.1038/nn.3101)); the graph measures follow the definitions of **Mikail
Rubinov and Olaf Sporns** ([paper](https://doi.org/10.1016/j.neuroimage.2009.10.003)); the
hemispheric summary takes the work of **Thibault Verhoeven, Pieter van Mierlo, Serge Vulliemoz
and colleagues** on network-based lateralisation of temporal lobe epilepsy
([paper](https://doi.org/10.1016/j.nicl.2017.09.021)) as its design reference. Pennsieve packages
these methods as an app; it did not develop them. The demo data are the **Siena Scalp EEG
Database** of Paolo Detti and colleagues at the University of Siena (BIDS conversion by Jonathan
Dan et al., EPFL), the **CHB-MIT Scalp EEG Database** of Ali Shoeb and John Guttag (MIT, with
Children's Hospital Boston), and **OpenNeuro ds003555** of Dorottya Cserpan, Johannes Sarnthein and
colleagues (University Hospital Zurich).

One step of the **Scalp EEG Interictal Biomarkers** workflow
(`eeg-prep → {eeg-spectral, eeg-connectivity, eeg-sleep → eeg-spikes} → eeg-report`).

> **Research output.** Connectivity values, node strengths, graph measures and the hemispheric
> asymmetry are quantitative EEG measures for research, to be reviewed by a clinician against the
> raw EEG. They are not a clinical EEG interpretation, and **no part of this app has been
> validated for diagnosis or for lateralising an epileptogenic zone**.

## What it does

**For the epileptologist.** Two scalp electrodes can show the same rhythm because they record one
brain region through the skull (volume conduction), or because two brain regions interact. Only
the second kind is a delay between the two signals. This app measures, for every pair of
channels and every frequency band, how consistently one channel leads or lags the other, which
volume conduction and the shared reference cannot produce. It then summarises which channels are
most connected (node strength), how the network is organised (graph measures), and whether the
left hemisphere is more or less connected internally than the right. Three figures per recording
show where to look: the connectivity matrix per band (ordered left | midline | right), a circle
plot of the strongest connections, and a scalp map of node strength; one overview figure shows
the hemispheric asymmetry of every recording. Seizures (plus 60 s on each side), annotated
artifacts and bad segments, and automatically detected artifacts are left out, so the numbers
describe the interictal background.

**For the researcher.** Sensor-space, volume-conduction-robust functional connectivity of
clinical scalp EEG on 4 s artifact-free epochs: wPLI (default), imaginary coherence and
orthogonalised amplitude envelope correlation per band (delta, theta, alpha, beta, gamma, as
eeg-spectral); node strength; weighted graph measures with NetworkX on proportionally thresholded
networks (clustering, characteristic path length, global efficiency, Louvain modularity,
small-worldness against degree-preserving null networks); hemispheric within/between means and
an asymmetry index. Everything is in long tables keyed by `recording_id` (and channel or channel
pair), plus the shared `features.csv` (`<measure>_<band>` node strength) that the iEEG apps and a
future normative step read. There is no validated open-source scalp-EEG connectivity classifier
for epilepsy (the published lateralisation results, e.g. Verhoeven et al. 2018, used
source-imaging pipelines that were not released); this app is a transparent sensor-level feature
extractor in that direction, not a reimplementation.

### Method

1. **Input.** The eeg-prep derivative (`recordings.csv`, `*_desc-preproc_eeg.edf`, `_channels.tsv`,
   `_events.tsv`), any BIDS-EEG dataset, or loose EDF/BDF/BrainVision/FIF/EEGLAB files, found
   recursively; the same reader as eeg-spectral (`EEG Fp1-Ref`, `Fp1-Avg` → `Fp1`; `T3/T4/T5/T6`
   → `T7/T8/P7/P8`; non-EEG channels and channels marked bad are left out; EDFs without a
   physical dimension, as in ds003555, are read as µV).
2. **Reference.** Referential recordings are analysed in **average reference** over the channels
   used (re-referenced in the app when the input is in another reference, with a note).
   **Bipolar-only recordings** (CHB-MIT, many clinical exports) cannot be turned back into
   electrodes, so connectivity is computed between the bipolar derivations, with a **warning**
   in `summary.json` and the log: neighbouring derivations share an electrode (Fp1-F7 and F7-T7
   share F7), so their signals are not independent. The three measures ignore the zero-lag part
   a shared electrode adds, but bipolar values are not comparable with referential ones.
3. **Epochs.** Non-overlapping 4 s epochs (`EPOCH_S`) over the analysable EEG: the recording minus
   `bad_segment`/`artifact` events and minus each annotated seizure with `SEIZURE_MARGIN_S` (60 s)
   before and after, read 60 s at a time.
4. **Automatic rejection** (`REJECT=true`), eeg-spectral's rules adapted to connectivity, which
   needs every channel in the same epoch: an epoch is left out when any channel exceeds
   `REJECT_UV` (500 µV) peak-to-peak or is flat (< 0.5 µV), or when its median 20–45 Hz power is
   a robust outlier (median + 4 × 1.4826 MAD over the recording's epochs: muscle); a channel flat
   or above `REJECT_UV` in most epochs is left out. At least 10 epochs must remain.
5. **wPLI and imaginary coherence** (MNE-Connectivity `spectral_connectivity_epochs`, multitaper
   with its default bandwidth: 7 DPSS tapers, ±1 Hz smoothing on 4 s epochs), averaged over the
   frequency bins of each band. wPLI (0–1) is |E[Im S<sub>xy</sub>]| / E[|Im S<sub>xy</sub>|] over
   epochs and tapers: 1 = the phase of one channel consistently leads the other. imcoh is the
   imaginary part of the coherency (signed, −1 to 1; the sign is the direction of the lag). The
   epochs are streamed through the estimator one at a time, so memory does not grow with the
   recording.
6. **Orthogonalised envelope correlation** (`aec`): each chunk of continuous analysable EEG is
   band-passed (4th-order Butterworth, zero phase) and Hilbert-transformed; MNE-Connectivity's
   `envelope_correlation(orthogonalize="pairwise")` removes from each signal the part that is
   in phase with the other before correlating their amplitude envelopes, over the kept epochs of
   the chunk; values are averaged over chunks weighted by their length.
7. **Node strength**: the mean |connectivity| of a channel with every other channel (strength /
   (N − 1), so recordings with different channel counts can be compared).
8. **Graph measures** (definitions of Rubinov & Sporns 2010, computed with NetworkX on |values|):
   the strongest `DENSITY` (20 %) of all channel pairs are kept and normalised to the strongest
   edge; `clustering` = mean weighted clustering coefficient (Onnela et al. 2005); `path_length` =
   characteristic path length with edge length 1/weight, over connected pairs;
   `global_efficiency` = mean inverse shortest weighted path over all pairs; `modularity` =
   Newman's Q of the Louvain partition (Blondel et al. 2008; seeded); `small_worldness` =
   (C / C<sub>null</sub>) / (L / L<sub>null</sub>) (Humphries & Gurney 2008) against `N_NULL` (20)
   null networks with the same degree sequence (double edge swaps, Maslov & Sneppen 2002) and
   the same weights shuffled over the rewired edges; `density` = kept / possible edges.
9. **Hemispheres**: `left` and `right` are the mean |connectivity| over pairs of channels within
   the left and within the right hemisphere (midline channels left out; for bipolar pairs, a pair
   whose two electrodes are on one side), `interhemispheric` over left-right pairs, and
   `asymmetry` = (left − right) / (left + right): positive = more connectivity within the left
   hemisphere. This is a **sensor-level proxy** of the source-level, directed network asymmetry
   that Verhoeven et al. 2018 used to lateralise temporal lobe epilepsy; it is not their method.

## Ports

| Port | Direction | Content |
|---|---|---|
| `eeg` | input (subfolder) | eeg-prep's BIDS-EEG derivative, any BIDS-EEG dataset, or loose EEG files |
| `connectivity` | output | `eeg-connectivity/` with the tables, `figures/`, `summary.json`, `provenance.json` |

## Parameters

| Name | Default | What it does |
|---|---|---|
| `MEASURES` | `wpli` | comma-separated `wpli`, `imcoh`, `aec`, or `all`; the first is shown in figures and headlines |
| `BANDS` | `conventions` | delta 1–4, theta 4–8, alpha 8–13, beta 13–30, gamma 30–45 Hz, or a JSON object `{"alpha": [8, 13], ...}` |
| `EPOCH_S` | `4` | epoch length (s); the lowest band edge must be at least 2 cycles per epoch |
| `DENSITY` | `0.2` | proportion of the strongest edges kept for the graph measures |
| `N_NULL` | `20` | degree-preserving null networks for small-worldness (0 = none) |
| `SEED` | `42` | seed of the rewiring and the Louvain partition |
| `TOP_EDGES` | `20` | edges per band in the circle plot |
| `EXCLUDE_SEIZURE` | `true` | leave annotated seizures out |
| `SEIZURE_MARGIN_S` | `60` | seconds left out before and after each seizure |
| `REJECT` | `true` | automatic artifact rejection (above) |
| `REJECT_UV` | `500` | peak-to-peak threshold per epoch (µV) |
| `MAX_EPOCHS` | `0` | 0 = all artifact-free epochs; otherwise this many, spread evenly over the recording |
| `FIGURES` | `true` | write the figures |
| `CHUNK_S` | `60` | seconds read at a time |
| `THREADS` | `0` | BLAS threads; 0 = from the container's CPU quota, capped by memory |
| `LABEL` | (empty) | write to `eeg-connectivity/<LABEL>/` |

## Outputs (under `OUTPUT_DIR/eeg-connectivity/`)

| File | Columns |
|---|---|
| `connectivity.csv` | recording_id, channel_a, channel_b, band, measure (`wpli`, `imcoh`, `aec`), value (each pair once) |
| `node_strength.csv` | recording_id, channel, band, measure, strength (mean \|value\| with every other channel) |
| `graph.csv` | recording_id, band, measure, metric (`clustering`, `path_length`, `global_efficiency`, `modularity`, `small_worldness`, `density`), value |
| `hemisphere.csv` | recording_id, band, measure, left, right, interhemispheric, asymmetry |
| `features.csv` | recording_id, channel, feature (`<measure>_<band>`, e.g. `wpli_alpha`: node strength), value |
| `figures/<recording_id>_{matrix,edges,strength}.png` | connectivity matrix per band; strongest edges per band (circle plot); node-strength scalp maps |
| `figures/asymmetry_overview.png` | hemispheric asymmetry per recording and band (when there is more than one recording) |
| `summary.json` | headline numbers per recording (`<measure>_<band>_mean`, `_asymmetry`, `_small_worldness`, the strongest alpha edge, epochs used), the figures with captions (in order), tables, warnings |
| `provenance.json` | app, version, tools, parameters, inputs, timings, peak memory, threads, per-recording status, montage, intervals and rejection counts, citations |

A recording that fails (unreadable, fewer than 4 EEG channels, sampling rate below 100 Hz, fewer
than 10 artifact-free epochs) is logged and listed as failed; the step fails only when every
recording fails or nothing is found (exit 1); bad parameters exit 2.

## Runtime and results on the demo data

Measured in Docker with **amd64 emulation on an M-series Mac** (so slower than on AWS), image
`eeg-connectivity` 0.1.0 (161 MB). One recording at a time, read twice in 60 s chunks; memory does
not grow with the length of the recording. Almost all of the time is the multitaper cross-spectra
and, with `aec`, the band-pass filters (three to four times the wPLI-only time).

| Input | Measures | Recordings | Wall time | Peak memory (cgroup) |
|---|---|---|---|---|
| Synthetic fixture, Lambda-style (`--cpus 1`, read-only, uid 65534) | all | 2 (6 and 4 min) | 21 s | 312 MB |
| eeg-prep's Siena derivative, sub-00 (`--cpus 2 --memory 3g`) | wpli | 5 runs, 35-44 min | 16 s | 332 MB |
| the same | all | 5 runs | 70 s | 332 MB |
| eeg-prep's Siena derivative, all of it | all | 41 runs, 0.6-13.5 h, 41 ok | 1760 s | 870 MB |
| one 13.5 h run of it (sub-01 run-00, 19 channels, 256 Hz) | all | 1 | 159 s | (within the above) |
| CHB-MIT chb01_01/03/04, raw bipolar EDF + summary | all | 3 x 1 h | 62 s | 342 MB |
| OpenNeuro ds003555 sub-01..03, 3 h sleep EEG at 1024 Hz | all | 3 | 422 s (89-185 s each) | 540 MB |

The cgroup peak includes the page cache of the EDF being read. The app is sized at **1 vCPU,
2 GB** with a **2 h** timeout: a step gets every recording of its input, and a dataset of
long-term recordings (the 41 Siena runs, 128 h of EEG) takes about half an hour under emulation
with all measures; a single routine EEG takes seconds.

What it found:

- **Synthetic fixture**: the planted 39 ms-lagged theta pair T7-O1 is the strongest theta edge
  for all three measures (wPLI 0.90, imcoh -0.70, aec 0.73), and the 11.7 ms-lagged beta pair
  F3-P7 the strongest beta edge (wPLI 0.63, imcoh -0.48, aec 0.69); the zero-lag copies stay at
  the noise level (wPLI 0.08-0.17, |imcoh| < 0.03, aec < 0.07); the hemispheric asymmetry of
  theta and beta is +0.21 to +0.83 (left), alpha and gamma within ±0.11; on the bipolar-only
  variant the strongest theta edge links a T7-containing and an O1-containing pair (wPLI 0.91).
  With `EXCLUDE_SEIZURE=false` the planted right-hemisphere seizure raises right theta wPLI, so
  leaving seizures out matters.
- **Siena** (eeg-prep derivative, 41 runs of 14 adults with focal epilepsy): 2 % of epochs
  rejected in the median run (24 % in the worst); median wPLI over all pairs 0.21 in theta and
  0.22 in alpha, 0.13-0.18 in the other bands; wPLI small-worldness 0.3-2.0 (median about 0.85-1.1
  by band), i.e. thresholded 19-channel sensor networks are not clearly small-world.
- **Exploratory lateralisation check on Siena** (not a validation): for the 13 patients whose
  annotated seizure channels are on one side (9 left, 4 right; sub-10 bilateral), the sign of the
  mean within-hemisphere asymmetry per patient matched the seizure side in 4-8 of 13 patients
  depending on measure and band (wPLI: delta 5, theta 4, alpha 4, beta 6, gamma 7; imcoh 4-8;
  aec 5-7): **chance level**. The sensor-level asymmetry does not lateralise these patients, and
  must not be read as a lateralising sign.
- **CHB-MIT chb01** (bipolar only): runs with the bipolar warning; 21 of 23 derivations used (the
  reversed duplicate `P7-T7` and the `T8-P8-1` copy left out); on these raw, unfiltered files 13-50 %
  of epochs were rejected (eye blinks in the frontal derivations; chb01_01 also lost Fp1-F7 and
  Fp1-F3 as noisy in most epochs). Run eeg-prep first on raw clinical data.
- **ds003555**: EDFs without a physical dimension are read as µV and re-referenced to the average
  (noted per recording); sub-01 lost Pz and 65 % of its epochs to artifacts.

Outputs on the demo data are in
`workflow-scalp-eeg/data/work/{siena,chbmit,ds003555}/eeg-connectivity/` (all measures).

## Limitations

- **Not validated for lateralisation or diagnosis.** The hemispheric asymmetry is a sensor-level
  proxy; the published network lateralisation of temporal lobe epilepsy (Verhoeven et al. 2018)
  used source imaging, directed connectivity and a trained classifier, which this app does not
  reproduce. On the Siena demo data the sign of the asymmetry matched the side of the annotated
  seizures only at chance level (above).
- **Sensor space.** A connection between two scalp channels is not a connection between the two
  brain regions under them: every channel mixes many sources. wPLI, imaginary coherence and the
  orthogonalised envelope correlation remove zero-lag mixing, but a single deep source seen with
  slightly different phases, or a lagged artifact (eye movements, ECG with a pulse delay), can
  still produce non-zero values. Run eeg-prep (ICA + ICLabel) first.
- **wPLI measures consistency, not size.** A tiny but consistent phase lag gives a high wPLI (the
  average reference spreads a little of every lagged pair to all channels, which shows up as weak
  wPLI between unrelated channels); imaginary coherence scales with the size of the lagged part.
  wPLI also has a positive bias with few epochs (about 0.1–0.15 with 50–60 epochs on the fixture):
  compare recordings with similar numbers of epochs (`epochs_used`).
- **State.** Long-term recordings are averaged over wake and sleep, which have very different
  networks. Compare like with like (`MAX_EPOCHS` evens the epoch count, not the state).
- **Reference and montage** change every number: compare recordings only in the same montage
  (`provenance.json` says which). Bipolar derivations share electrodes (warning above).
- Graph measures on 19 scalp channels are coarse and depend on the threshold (`DENSITY`) and the
  measure; small-worldness of a thresholded sensor network says little about the brain's
  small-world organisation.
- The envelope correlation is computed on 4 s epochs grouped by 60 s chunk; slow (delta) envelopes
  are poorly sampled by short segments, so delta `aec` is noisy.

## Licences

Our code: Apache-2.0 (`LICENSE`). Bundled components and their licences: `THIRD_PARTY.md`
(MNE-Connectivity BSD-3-Clause, NetworkX BSD-3-Clause, MNE-Python BSD-3-Clause, NumPy/SciPy/pandas/
scikit-learn BSD-3-Clause, xarray Apache-2.0, Matplotlib PSF-based; all permissive). bctpy is
GPL-3.0 and is not used. The demo databases are not bundled: Siena CC BY 4.0 (BIDS conversion;
PhysioNet original ODC-By 1.0), CHB-MIT ODC-By 1.0, ds003555 CC0.

## Citations

Please cite these works if you use this app. Verified on Crossref/DataCite and PubMed; full author
lists in `CITATION.cff`.

- Vinck M, Oostenveld R, van Wingerden M, Battaglia F, Pennartz CMA. An improved index of
  phase-synchronization for electrophysiological data in the presence of volume-conduction, noise
  and sample-size bias. *NeuroImage* 55(4):1548–1565 (2011). doi:10.1016/j.neuroimage.2011.01.055,
  PMID 21276857 (10.1016/j.neuroimage.2011.01.055).
- Nolte G, Bai O, Wheaton L, Mari Z, Vorbach S, Hallett M. Identifying true brain interaction from
  EEG data using the imaginary part of coherency. *Clinical Neurophysiology* 115(10):2292–2307
  (2004). doi:10.1016/j.clinph.2004.04.029, PMID 15351371 (10.1016/j.clinph.2004.04.029).
- Hipp JF, Hawellek DJ, Corbetta M, Siegel M, Engel AK. Large-scale cortical correlation structure
  of spontaneous oscillatory activity. *Nature Neuroscience* 15(6):884–890 (2012).
  doi:10.1038/nn.3101, PMID 22561454 (10.1038/nn.3101).
- Binns TS, Li A, Larson E, McCloy D, Rockhill A, Li Q, Ruuskanen S, Kroner A, Gramfort A,
  Marraffini GFG, Shor J, Marshall K, Orabe M, Gerster M, Stevenson RF, Köhler RM, Steingold S,
  Mert S. MNE-Connectivity (version 0.9.0). Zenodo. doi:10.5281/zenodo.10278399.
- Rubinov M, Sporns O. Complex network measures of brain connectivity: Uses and interpretations.
  *NeuroImage* 52(3):1059–1069 (2010). doi:10.1016/j.neuroimage.2009.10.003, PMID 19819337
  (10.1016/j.neuroimage.2009.10.003).
- Verhoeven T et al. Automated diagnosis of temporal lobe epilepsy in the absence of interictal
  spikes. *NeuroImage: Clinical* 17:10–15 (2018). doi:10.1016/j.nicl.2017.09.021, PMID 29527470
  (10.1016/j.nicl.2017.09.021).
- Gramfort A et al. MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience* 7:267
  (2013). doi:10.3389/fnins.2013.00267, PMID 24431986 (10.3389/fnins.2013.00267).
- Hagberg AA, Schult DA, Swart PJ. Exploring Network Structure, Dynamics, and Function using
  NetworkX. *Proceedings of the 7th Python in Science Conference (SciPy 2008)*, 11–15 (2008).
  doi:10.25080/TCWV9851.
- Blondel VD et al. Fast unfolding of communities in large networks. *Journal of Statistical
  Mechanics: Theory and Experiment* 2008(10):P10008 (2008). doi:10.1088/1742-5468/2008/10/P10008.
- Onnela J-P et al. Intensity and coherence of motifs in weighted complex networks. *Physical
  Review E* 71(6):065103 (2005). doi:10.1103/PhysRevE.71.065103, PMID 16089800
  (10.1103/PhysRevE.71.065103).
- Maslov S, Sneppen K. Specificity and Stability in Topology of Protein Networks. *Science*
  296(5569):910–913 (2002). doi:10.1126/science.1065103, PMID 11988575 (10.1126/science.1065103).
- Humphries MD, Gurney K. Network 'Small-World-Ness': A Quantitative Method for Determining
  Canonical Network Equivalence. *PLoS ONE* 3(4):e0002051 (2008). doi:10.1371/journal.pone.0002051,
  PMID 18446219 (10.1371/journal.pone.0002051).

Demo data:

- Detti P. Siena Scalp EEG Database (version 1.0.0). PhysioNet (2020). doi:10.13026/5d4a-j060;
  Detti P et al. EEG Synchronization Analysis for Seizure Prediction: A Study on Data of
  Noninvasive Recordings. *Processes* 8(7):846 (2020). doi:10.3390/pr8070846; Dan J, Detti P.
  BIDS Siena Scalp EEG Database (v1.0.0). EPFL / Zenodo (2024). doi:10.5281/zenodo.10640762.
- Shoeb A. CHB-MIT Scalp EEG Database. PhysioNet (2010). doi:10.13026/C2K01R; Shoeb AH.
  Application of machine learning to epileptic seizure onset detection and treatment. PhD thesis,
  MIT (2009). https://dspace.mit.edu/handle/1721.1/54669.
- Goldberger AL et al. PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 101(23):e215
  (2000). doi:10.1161/01.CIR.101.23.e215, PMID 10851218 (10.1161/01.CIR.101.23.e215).
- Cserpan D, Boran E, Rosch R, Lo Biundo SP, Ramantani G, Sarnthein J. Dataset of EEG recordings
  of pediatric patients with epilepsy based on the 10-20 system. OpenNeuro (2021).
  doi:10.18112/openneuro.ds003555.v1.0.1; Cserpan D et al. Scalp high-frequency oscillation rates
  are higher in younger children. *Brain Communications* 3(2):fcab052 (2021).
  doi:10.1093/braincomms/fcab052, PMID 33870193 (10.1093/braincomms/fcab052).

## Acknowledgements

MNE-Connectivity and MNE-Python are developed by the MNE community; NetworkX by the NetworkX
developers. MNE-Connectivity asks users to cite its Zenodo DOI and NetworkX asks for Hagberg et
al. 2008, both above. The Siena Scalp EEG Database was recorded at the Unit of Neurology and
Neurophysiology of the University of Siena and published on PhysioNet; the CHB-MIT database was
collected at Children's Hospital Boston and published on PhysioNet. PhysioNet asks users to cite
Goldberger et al. 2000 together with each database.

## Development

    python3.12 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
    make test PYTHON=.venv/bin/python        # pytest on the synthetic fixture, no Docker
    make run                                  # Docker over data/fixture/eeg -> data/output/ (all measures, dev.env)
    make lambda-check                         # read-only image, uid 65534, HOME/TMPDIR under /tmp
    make run-demo DEMO=../workflow-scalp-eeg/data/work/siena/eeg-prep/eeg

The fixture (`tests/make_fixture.py`, also the release review's input) plants, on a 1/f
background: a 5–7 Hz source on T7 repeated 39 ms later on O1 and a 16–24 Hz source on F3 repeated
11.7 ms later on P7 (phase-lagged coupling, left hemisphere); zero-lag copies of other 5–7 Hz and
16–24 Hz sources on F8/T8/F4 and C4/P4 (volume conduction, right hemisphere); a zero-lag posterior
alpha rhythm; a bad segment, an electrode pop, a bad channel and a seizure that spreads over the
right hemisphere with a lag; in a referential (average reference) and a bipolar-only recording,
plus a Siena-style plain BIDS dataset and a loose CHB-MIT-style EDF. The tests assert that the
lagged pairs are the strongest edges for all three measures, that the zero-lag copies stay low,
that the asymmetry points left, that the seizure is left out (and would raise right theta wPLI if
it were not), and the table shapes, exit codes and warnings.
