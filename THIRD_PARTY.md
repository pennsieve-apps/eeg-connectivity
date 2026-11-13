# Third-party components

The Docker image bundles these Python packages (exact versions in `requirements.txt`), installed
from PyPI wheels on `python:3.12.16-slim-bookworm` (Debian bookworm with CPython 3.12, PSF licence).
No model weights or datasets are bundled: every measure is computed from the data, and the 10-05
electrode template positions ship inside MNE-Python. bctpy (GPL-3.0) is **not** used: the graph
measures are computed with NetworkX.

| Component | Version | Authors / copyright holder | Licence | URL |
|---|---|---|---|---|
| mne-connectivity | 0.9.0 | Thomas S. Binns, Adam Li, Eric Larson and the MNE-Connectivity contributors (MNE tools for MEG and EEG data analysis) | BSD-3-Clause | https://github.com/mne-tools/mne-connectivity |
| NetworkX | 3.7 | Aric Hagberg, Dan Schult, Pieter Swart and the NetworkX Developers | BSD-3-Clause | https://github.com/networkx/networkx |
| MNE-Python | 1.13.2 | Alexandre Gramfort and the MNE-Python contributors | BSD-3-Clause | https://github.com/mne-tools/mne-python |
| NumPy | 2.5.4 | NumPy Developers | BSD-3-Clause (bundled parts: 0BSD, MIT, Zlib, CC0-1.0) | https://numpy.org |
| SciPy | 1.18.1 | SciPy Developers | BSD-3-Clause | https://scipy.org |
| Matplotlib | 3.11.2 | Matplotlib Development Team | Matplotlib License (PSF-based) | https://matplotlib.org |
| pandas | 3.1.1 | The pandas development team | BSD-3-Clause | https://github.com/pandas-dev/pandas |
| xarray | 2026.9.0 | xarray Developers | Apache-2.0 | https://github.com/pydata/xarray |
| scikit-learn | 1.9.1 | The scikit-learn developers | BSD-3-Clause | https://github.com/scikit-learn/scikit-learn |
| joblib | 1.6.0 | joblib developers (Gael Varoquaux and contributors) | BSD-3-Clause | https://github.com/joblib/joblib |
| threadpoolctl | 3.8.0 | joblib developers | BSD-3-Clause | https://github.com/joblib/threadpoolctl |
| cloudpickle | 3.1.2 | The cloudpickle developer team | BSD-3-Clause | https://github.com/cloudpipe/cloudpickle |
| narwhals | 2.28.0 | Marco Gorelli and narwhals contributors | MIT | https://github.com/narwhals-dev/narwhals |
| tzdata | 2026.6 | Python Software Foundation (IANA time zone data: public domain) | Apache-2.0 | https://github.com/python/tzdata |
| contourpy | 1.4.0 | ContourPy Developers | BSD-3-Clause | https://github.com/contourpy/contourpy |
| cycler | 0.12.1 | Matplotlib project | BSD-3-Clause | https://github.com/matplotlib/cycler |
| decorator | 5.3.1 | Michele Simionato | BSD-2-Clause | https://github.com/micheles/decorator |
| fonttools | 4.68.0 | Just van Rossum and fontTools contributors | MIT | https://github.com/fonttools/fonttools |
| Jinja2 | 3.1.6 | Pallets | BSD-3-Clause | https://github.com/pallets/jinja |
| kiwisolver | 1.5.1 | Nucleic Development Team | BSD-3-Clause | https://github.com/nucleic/kiwi |
| lazy-loader | 0.6 | Scientific Python | BSD-3-Clause | https://github.com/scientific-python/lazy-loader |
| MarkupSafe | 3.0.4 | Pallets | BSD-3-Clause | https://github.com/pallets/markupsafe |
| packaging | 26.3 | Donald Stufft and contributors | Apache-2.0 OR BSD-2-Clause | https://github.com/pypa/packaging |
| Pillow | 13.0.0 | Jeffrey A. Clark and contributors | MIT-CMU | https://github.com/python-pillow/Pillow |
| platformdirs | 4.14.5 | platformdirs contributors | MIT | https://github.com/tox-dev/platformdirs |
| pooch | 1.9.0 | The Pooch Developers | BSD-3-Clause | https://github.com/fatiando/pooch |
| pyparsing | 3.3.3 | Paul McGuire | MIT | https://github.com/pyparsing/pyparsing |
| python-dateutil | 4.0.0 | dateutil contributors | BSD-3-Clause | https://github.com/dateutil/dateutil |
| requests | 2.34.2 | Kenneth Reitz and contributors | Apache-2.0 | https://github.com/psf/requests |
| certifi | 2026.10.23 | Kenneth Reitz | MPL-2.0 | https://github.com/certifi/python-certifi |
| charset-normalizer | 3.5.2 | Ahmed Tahri | MIT | https://github.com/jawah/charset_normalizer |
| idna | 3.20 | Kim Davies | BSD-3-Clause | https://github.com/kjd/idna |
| urllib3 | 2.8.0 | Andrey Petrov and contributors | MIT | https://github.com/urllib3/urllib3 |
| six | 1.17.0 | Benjamin Peterson | MIT | https://github.com/benjaminp/six |
| tqdm | 4.70.1 | tqdm developers | MPL-2.0 AND MIT | https://github.com/tqdm/tqdm |

pandas, xarray, scikit-learn (with joblib, threadpoolctl, cloudpickle, narwhals) and tzdata are
installed because mne-connectivity imports them; this app does not call them directly.
mne-connectivity also declares netCDF4 (MIT) and cftime (MIT) for saving connectivity objects to
netCDF files; this app never saves them, so they are not installed.

Test-only (not in the image): pytest (MIT), edfio 0.4.18 (Apache-2.0, writes the synthetic fixture's EDF files).

The methods implemented in this app's own code (epoch selection and rejection, copied from
eeg-spectral; band-pass and Hilbert transform before the envelope correlation; proportional
thresholding, the null-network comparison and the hemispheric summaries) were written from the
papers cited in the README; no code was copied from their authors.
