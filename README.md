Requires a Hugging Face token to access SAM3.

Environment setup:

```bash
conda env create -f environment.yml
conda activate flowsis
pip install -e .
```

Detector commands use a common interface:

```bash
# RT-DETRv2 defaults
flowsis-train-detector

# Architecture defaults select the matching model and output directory
flowsis-train-detector --detector dfine

# A checkpoint or model id can identify its own architecture
flowsis-live-detector --model outputs/detectors/dfine/final
flowsis-evaluate-detector --model outputs/detectors/dfine/final

# Online base-head training with a frozen detector
flowsis-train-base-head --detector dfine
flowsis-train-base-head --detector-model outputs/detectors/dfine/final
```

The detector API follows the same rule:

```python
from flowsis.pretrained import load_detector

detector = load_detector()  # RT-DETRv2 default
detector = load_detector(architecture="dfine")  # D-FINE default
detector = load_detector("outputs/detectors/dfine/final")  # inferred
```

Detector checkpoints contain `flowsis_detector.json`, and base-head checkpoints
are versioned `head.pt` bundles containing both weights and architecture
configuration. Cached detector features use the versioned
`feature_bundle.pt` interface in `flowsis.data`; cache generation and offline
augmentation are intentionally not implemented yet.

PTLFlow note:

`ptlflow==0.4.2` is installed from PyPI in [environment.yml](/home/pricyspark/FlowSIS/environment.yml). As of July 7, 2026, PyPI includes `0.4.2`, and it requires Python `>=3.8,<3.14`, so the environment pins Python `3.13` to avoid Conda selecting Python `3.14`, which `pip` will reject for PTLFlow.

For development, the environment keeps most packages on `conda-forge` and leaves only `ptlflow`, `opencv-python`, `kaleido`, and `kernels` on `pip`. `kernels` stays pinned to `0.14.1` for the Hugging Face issue you hit, and `opencv-python` stays on the `pip` side intentionally, because mixing Conda OpenCV with PTLFlow's `pip` dependency resolution caused uninstall conflicts during environment creation.
