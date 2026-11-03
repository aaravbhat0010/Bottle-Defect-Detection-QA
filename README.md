# Bottle Defect Detection with Automated QA

PyTorch convolutional autoencoder baseline using real MVTec AD bottle photos,
with automated CPU tests. This is an educational QA project with weak detection performance.

## Experiment

167 normal images train the model; 42 normal validation images select the best
checkpoint and a 95th-percentile reconstruction-error cutoff. All 83 official test
images (20 normal, 63 defective) remain outside training and threshold selection.
Seed 42; RGB 128×128; pixel values divided by 255; batch size 16; Adam learning
rate 0.001; 20 epochs; mean squared reconstruction loss.

Encoder: three stride-2 Conv2d layers, channels 3→16→32→64, with ReLU.
Decoder: three ConvTranspose2d layers, channels 64→32→16→3, ReLU between
layers and final sigmoid. The compressed representation is 64×16×16.
The anomaly score is whole-image mean squared reconstruction error, not a probability.

## Actual original Colab results

These are recovered saved outputs, not a new training or evaluation run.

| Metric | Value |
|---|---:|
| ROC-AUC | 0.6087301587 |
| Precision | 0.7954545455 |
| Recall | 0.5555555556 |
| F1 | 0.6542056075 |
| Threshold | 0.002495901833754033 |

| Actual / Predicted | Normal | Defective |
|---|---:|---:|
| Normal | 11 | 9 |
| Defective | 28 | 35 |

The baseline missed 28 defects and produced 9 false alarms. See results/original_*.txt
for the original saved training, threshold, evaluation and test output.
Inspected examples: a caught broken bottle (error 0.00256), missed broken bottle
(0.00220), false alarm on a good bottle (0.00270), and accepted good bottle (0.00228).
These rounded scores illustrate overlap between normal and defect scores. Dataset
images are excluded; rerun the notebook locally to view prediction examples.

## Setup

Python 3.12 is recommended. From this repository's root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pytest tests -q
```

Windows activation: `.venv\Scripts\activate`.
PyTorch supplies tensors, neural network layers and gradients. NumPy handles arrays;
Pillow reads images; scikit-learn calculates metrics; Matplotlib displays comparisons;
pytest runs software tests. Dependency ranges are in requirements.txt. The original
notebook reported PyTorch 2.11.0+cpu; recovery verification uses the separately recorded
environment in results/recovery_checks.txt. This is not an exact dependency lock.

## Get data and reproduce

Visit https://www.mvtec.com/research-teaching/datasets/mvtec-ad , review its terms,
and follow the official download form/instructions. Extract the bottle category locally.
The root must contain train/good, test/good, and test defect subfolders.
Do not add dataset images, masks, archives or checkpoints to GitHub.

```bash
python -m bottle_qa.train --root data/bottle
```

Adjust the root to the extracted folder. This recovered script runs the original split,
training, normal-validation calibration and official-test evaluation. It writes the best
checkpoint and threshold under artifacts/. Full training is not part of CI.
The cleaned notebook in notebooks/Bottle_Defect_QA.ipynb provides the original
interactive Colab workflow, including upload and prediction displays. Its outputs and
embedded images have been removed. CPU is the default; no GPU is required.
The original notebook is preserved separately by its owner.

## Single-image inference

After training locally:

```python
import json
from bottle_qa.inference import predict_image
threshold = json.load(open("artifacts/threshold.json"))["threshold"]
print(predict_image("data/bottle/test/good/000.png",
                    "artifacts/bottle_autoencoder.pt", threshold))
```

Only load trusted checkpoints. The historical threshold is valid for the original model;
use the newly calibrated threshold after retraining.

## Software QA

The 11 recovered cases check: red-image shape/type/finite values/exact channel scaling;
corrupt file rejection; model output shape/range; finite training loss and gradients and
changed weights; identical save/reload predictions; four category labels; complete
image/checkpoint/scoring/decision inference; and invalid threshold rejection.
Synthetic fixtures and temporary models keep CPU tests fast and dataset independent.
They do not establish useful detection quality or test every possible failure mode.

```bash
python -m pytest tests --collect-only -q
python examples/bug_demo.py
```

The separate bug demo removes division by 255, observes the red-image test fail,
restores scaling, and observes it pass. It uses temporary copies, leaving the main
loader unchanged. GitHub Actions runs tests on pushes and pull requests. Hosted
Actions execution is not verified merely by local success.

## Limitations

The reconstructions lose fine rim detail. Whole-image averaging can dilute localized
defects, while harmless appearance differences cause false alarms. Autoencoders can
also reconstruct defects. Low reconstruction error does not imply high detection accuracy.
Validation has only 42 normal images and is reused for checkpoint selection and
threshold calibration. The 95th-percentile rule does not guarantee 5% false alarms on
new data. Only one category, configuration and seed were evaluated. The notebook
checks path separation, not all possible near-duplicate image leakage.
Do not tune this baseline against test results and claim another independent evaluation:
changes inspired by this test set are exploratory and need fresh held-out data.
This project is not production ready.

## Interview explanation

I trained a simple PyTorch autoencoder on normal industrial bottle images and used
reconstruction error to flag anomalies. I separated training, validation and official
test data, measured real detection failures, and built automated tests for the pipeline.
A deliberate missing-normalization bug demonstrated why known synthetic fixtures help.
Software tests verify expected code behavior; held-out ML evaluation measures whether
predictions generalize. Passing tests did not make this weak detector reliable.

## Dataset attribution

MVTec AD, MVTec Software GmbH. Paul Bergmann, Michael Fauser, David Sattlegger,
Carsten Steger, “MVTec AD — A Comprehensive Real-World Dataset for Unsupervised
Anomaly Detection,” CVPR 2019.

Official page: https://www.mvtec.com/research-teaching/datasets/mvtec-ad
The official terms checked during the project list CC BY-NC-SA 4.0 and prohibit commercial
use of the data. License: https://creativecommons.org/licenses/by-nc-sa/4.0/ .
Dataset and masks are not redistributed in this package.
