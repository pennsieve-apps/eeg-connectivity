FROM --platform=linux/amd64 python:3.12.16-slim-bookworm@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337

ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /processor

# Pinned Python dependencies, binary wheels only, no dependency resolution at build time
# (requirements.txt lists every package). Test folders and parts of MNE this app never reads are
# removed (mne/data/fsaverage stays: the 10-05 template montage needs its fiducials). No model
# weights: every measure is computed from the data.
COPY requirements.txt /processor/requirements.txt
RUN pip install --no-cache-dir --no-deps --only-binary=:all: --root-user-action=ignore -r /processor/requirements.txt \
    && find /usr/local/lib/python3.12 -depth -type d -name __pycache__ -exec rm -rf {} + \
    && find /usr/local/lib/python3.12/site-packages -depth -type d \( -name tests -o -name testing \) -path "*/mne/*" -exec rm -rf {} + \
    && find /usr/local/lib/python3.12/site-packages -depth -type d -name tests \
            \( -path "*/mne_connectivity/*" -o -path "*/networkx/*" -o -path "*/xarray/*" -o -path "*/pandas/*" -o -path "*/sklearn/*" \) -exec rm -rf {} + \
    && rm -rf /usr/local/lib/python3.12/site-packages/scipy/*/tests /usr/local/lib/python3.12/site-packages/numpy/*/tests \
              /usr/local/lib/python3.12/site-packages/matplotlib/tests \
              /usr/local/lib/python3.12/site-packages/mne/html_templates/report \
    && python3 -c "import mne, mne_connectivity, networkx, matplotlib; mne.channels.make_standard_montage('colin27_1005'); from mne_connectivity.viz import plot_connectivity_circle; print('mne', mne.__version__, 'mne-connectivity', mne_connectivity.__version__, 'networkx', networkx.__version__)" \
    && mkdir -p /opt/eeg-connectivity/matplotlib \
    && MPLCONFIGDIR=/opt/eeg-connectivity/matplotlib python3 -c "import matplotlib.font_manager" \
    && ls /opt/eeg-connectivity/matplotlib/fontlist-*.json \
    && chmod -R a+rX /opt/eeg-connectivity

COPY processor/ /processor/processor/
COPY VERSION /processor/VERSION

# The Matplotlib font cache is built above into a read-only path; at start the app copies it
# to $TMPDIR/mpl (MPLCONFIGDIR) because HOME and /tmp are replaced at run time on Lambda.
# Threads (OMP/OpenBLAS/MKL) are set by the app from the cgroup CPU quota before numpy loads.
ENV PYTHONPATH="/processor" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    EEG_CONNECTIVITY_MPL_CACHE=/opt/eeg-connectivity/matplotlib \
    MPLBACKEND=Agg \
    MNE_DONTWRITE_HOME=true \
    MNE_USE_CUDA=false \
    MNE_DATA=/opt/eeg-connectivity/mne_data

# uid 1000 on ECS; any uid works (nothing in the image is written at run time).
RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin app
USER 1000

ENTRYPOINT ["python3", "-m", "processor.main"]
