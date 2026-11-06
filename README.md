# Deformation-Aligned Memory for Cardiac Cine MRI

CPU demo accompanying *Deformation-Aligned Memory for Interactive Segmentation of Cardiac Cine MRI*. Start with a point-prompted end-diastolic seed, follow the sequence through systole, and inspect how the decoder retrieves earlier features in physical coordinates.

## Quick start

From the repository root, run the following in a Python environment with pip:

```sh
python -m pip install -r requirements.txt
python check_read_interface.py
python run_demo.py --output outputs
python -m unittest discover -s tests -v
```

The interface check exercises the millimetre-based sampling and reliability gate used in the paper. A 2 mm displacement across an 8 mm slice spacing samples a quarter of the way to the next slice. With full support and zero round-trip residual the spatial weight is approximately 0.7311.

The training demo uses a small CPU network, a translation-only registration module, and a continuous reliability approximation (1.5 mm). The separate interface check implements the paper's thresholded gate (2 mm). The training script writes a toy checkpoint, end-systolic predictions and a metrics file. Its scores describe the synthetic test sequence. The full study model and its clinical evaluations are outside the scope of this demo.

## Data

- [Download the bundled synthetic sample](https://raw.githubusercontent.com/liyanan010401-wq/dam-cardiac-cine-demo/main/data/sample.npz). It contains five sequences with four phases each, geometric labels, ED points, an affine, and a fixed train/dev/test split. It can be regenerated with `python -m src.data`. See the [data card](data/DATA_CARD.md).
- [ACDC official dataset](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html). The [official download collection](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb) provides cardiac MRI and reference labels.
- [M&Ms official challenge](https://www.ub.edu/mnms/). Use the original challenge’s [access form](https://www.ub.edu/mnms/download.php) to request the images from the organizers.

All inputs required to run the demo are included. The clinical datasets are distributed by their respective providers.

[中文说明](README.zh-CN.md) · [Method-to-code map (中文)](METHOD_MAP.md)
