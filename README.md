# Bottle Defect Detection with Automated QA

A PyTorch project exploring industrial bottle defect detection with a convolutional autoencoder and automated software tests.

## Project Goal

Learn what normal bottles look like, use reconstruction error to flag possible defects, and test the software behind the workflow.

This project combines a real machine learning experiment with software QA. The baseline has limited detection performance, which is documented below.

## How It Works

1. Load a bottle photo.
2. Resize it to 128 × 128 pixels and convert it to RGB.
3. Scale pixel values from 0–255 to 0–1.
4. Pass the image through an encoder that compresses its information.
5. Use a decoder to reconstruct the image.
6. Calculate mean squared error between the reconstruction and its original.
7. Flag a possible defect if that error exceeds a fixed threshold.

The model learns only from normal training images. A high reconstruction error suggests an unusual image; it is not a calibrated probability of damage.

## Dataset

This project uses the bottle category from the official [MVTec AD dataset](https://www.mvtec.com/research-teaching/datasets/mvtec-ad).

The photos show bottles viewed from above. Defect categories include large breaks, small breaks, and contamination.

| Split | Images | Purpose |
|---|---:|---|
| Training | 167 normal | Update model weights |
| Validation | 42 normal | Select the checkpoint and threshold |
| Official test | 20 normal and 63 defective | Evaluate detection performance |

The 209 official normal training images were divided into training and validation using a fixed seed. The official test set was excluded from training, checkpoint selection, and threshold selection.

Ground truth masks identify defect locations, but this baseline evaluates whole images rather than pixel-level localization.

Dataset images, masks, and archives are not included in this repository.

## Model Architecture

The convolutional autoencoder has two parts:

| Part | Layers | Purpose |
|---|---|---|
| Encoder | Conv2d: 3 → 16 → 32 → 64 channels | Compress image information |
| Decoder | ConvTranspose2d: 64 → 32 → 16 → 3 channels | Reconstruct the image |

The encoder reduces spatial dimensions from 128 × 128 to 16 × 16. The compressed representation has shape 64 × 16 × 16.

ReLU activations are used between layers. A final sigmoid keeps reconstructed pixel values between 0 and 1.

## Training Configuration

| Setting | Value |
|---|---|
| Input | RGB, 128 × 128 |
| Normalization | Divide pixel values by 255 |
| Loss | Mean squared error |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 16 |
| Epochs | 20 |
| Random seed | 42 |
| Checkpoint selection | Lowest validation reconstruction error |
| Anomaly score | Mean squared reconstruction error per image |
| Threshold | 95th percentile of normal validation scores |

Training updates weights after each batch. Validation calculates reconstruction errors without updating weights.

Fixed seeds improve repeatability, but identical results across different hardware and software versions are not guaranteed.

## Actual Baseline Results

The following results were measured in my Colab run on the 83 official test images.

| Metric | Result |
|---|---:|
| Image-level ROC-AUC | 0.6087 |
| Precision | 0.7955 |
| Recall | 0.5556 |
| F1 score | 0.6542 |
| Reconstruction error threshold | 0.0024959 |

### Confusion Matrix

Rows show actual labels. Columns show predicted labels.

| | Predicted normal | Predicted defective |
|---|---:|---:|
| Actual normal | 11 | 9 |
| Actual defective | 28 | 35 |

The model:

- Correctly flagged 35 defective bottles.
- Missed 28 defective bottles.
- Correctly accepted 11 normal bottles.
- Incorrectly flagged 9 normal bottles.

Precision measures how many flagged bottles were actually defective. Recall measures how many actual defects were caught.

ROC-AUC uses continuous error scores to measure ranking performance across possible thresholds. It is not an accuracy percentage.

### Prediction Examples

Examples inspected in the notebook included:

| Actual category | Reconstruction error | Prediction | Outcome |
|---|---:|---|---|
| Large break | 0.00256 | Defective | Defect caught |
| Large break | 0.00220 | Normal | Defect missed |
| Good | 0.00270 | Defective | False alarm |
| Good | 0.00228 | Normal | Normal accepted |

These scores are rounded for display. Predictions use the full threshold value.

The examples show that a normal bottle can have a higher reconstruction error than a defective bottle. The underlying dataset images remain outside this repository.

## Automated QA Tests

My Colab test run passed 11 pytest cases.

| Test | What it verifies |
|---|---|
| Red image preprocessing | Correct image shape, float32 type, finite values, and exact RGB scaling |
| Corrupt image rejection | A file containing invalid image data is rejected |
| Model output | Correct output shape, finite values, and pixel range |
| Learning step | Finite loss and gradients, with weights changing after an update |
| Save and reload | Predictions remain identical after restoring saved weights |
| Good label | The good category maps to normal |
| Large break label | The large break category maps to defective |
| Small break label | The small break category maps to defective |
| Contamination label | The contamination category maps to defective |
| Complete prediction | Image loading, checkpoint loading, scoring, and threshold decisions work together |
| Invalid threshold | An invalid cutoff is rejected |

The label test runs four parameterized cases.

Tests use small synthetic images and temporary models. They require neither the MVTec dataset nor full model training.

Passing these tests verifies specific software behaviors. It does not establish useful defect detection performance.

## Deliberate Bug Demonstration

A separate example demonstrated how testing catches a preprocessing bug:

1. Create a synthetic red image with RGB values of 255, 0, 0.
2. Confirm the loader returns normalized values of 1, 0, 0.
3. Remove division by 255 from a separate loader copy.
4. Run the same test and observe a failure because the red value is 255 instead of 1.
5. Restore normalization and rerun the test successfully.

The main loader and trained model were not modified by this demonstration.

## Setup

Use Python 3.12. Run these commands from the repository folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

On Windows, replace the activation command with:

```powershell
.venv\Scripts\activate
```

### Run Tests

```bash
python -m pytest tests -q
```

### List Tests

```bash
python -m pytest tests --collect-only -q
```

### Dependencies

| Dependency | Purpose |
|---|---|
| PyTorch | Neural network layers, tensors, and automatic gradients |
| NumPy | Image arrays and numerical calculations |
| Pillow | Image loading, conversion, and resizing |
| scikit-learn | Evaluation metrics |
| Matplotlib | Image comparisons and visualizations |
| pytest | Automated software checks |

## Project Structure

```text
bottle_qa/
    __init__.py
    data.py
    model.py
    inference.py
tests/
    test_data.py
    test_model.py
    test_workflow.py
results/
    baseline.json
    environment.json
    test_run.txt
.github/
    workflows/
        tests.yml
.gitignore
requirements.txt
README.md
```

Training and evaluation were performed in the original Colab notebook. The source package contains preprocessing, architecture, inference, tests, and recorded results. It does not yet include a standalone training command.

## Single Image Inference

After creating a trained checkpoint locally:

```python
from bottle_qa.inference import predict_image

result = predict_image(
    image_path="data/bottle/test/good/000.png",
    checkpoint_path="artifacts/bottle_autoencoder.pt",
    threshold=0.002495901833754033,
)

print(result)
```

Adjust the image path to match the local dataset layout. The checkpoint and dataset are not distributed with this repository.

## Continuous Integration

The GitHub Actions workflow is configured to run CPU tests on every push and pull request.

It installs dependencies and runs:

```bash
python -m pytest tests -q
```

The workflow uses synthetic fixtures and temporary models. No dataset download, GPU, or full training run is required.

The successful Colab test run does not confirm that the hosted GitHub Actions workflow has passed. Its status must be checked separately in the repository's Actions tab.

## Limitations

- Reconstructions capture broad shapes but lose fine bottle rim details.
- Whole-image average error can dilute small, localized defects.
- Harmless lighting and texture differences can produce false alarms.
- An autoencoder can reconstruct defects and therefore miss them.
- Low reconstruction loss does not guarantee strong anomaly detection.
- Only one category, one seed, and one training configuration were evaluated.
- The 42 normal validation images were used for both checkpoint selection and threshold calibration.
- Three validation images exceeded the selected threshold. This does not guarantee a similar false alarm rate on new images.
- The measured detection performance is weak and does not support factory deployment.

The baseline threshold was fixed before test evaluation. Later changes motivated by these test results should be described as exploratory and evaluated on fresh held-out data.

## What I Learned

- How images become tensors and why preprocessing must stay consistent.
- How an autoencoder learns through reconstruction loss.
- How training, validation, and test sets serve different purposes.
- Why model evaluation and software testing answer different questions.
- How automated tests catch preprocessing and model workflow bugs.
- Why honest failure analysis matters when evaluating machine learning systems.

## Interview Summary

I built a PyTorch convolutional autoencoder to explore industrial bottle anomaly detection using real MVTec AD images. I trained on normal bottles, selected a threshold using normal validation images, and evaluated the fixed model on the official test set.

The baseline caught 35 of 63 defects and produced 9 false alarms. I documented these limitations rather than treating low reconstruction error as proof of good detection.

I also implemented 11 automated pytest cases covering preprocessing, invalid inputs, model outputs, gradients, weight updates, checkpoint consistency, labels, and inference. A separate bug demonstration showed that the tests catch missing pixel normalization.

## Dataset Attribution

Dataset: MVTec AD, MVTec Software GmbH.

Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger.  
*MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.*  
IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019.

Official dataset page:  
https://www.mvtec.com/research-teaching/datasets/mvtec-ad

MVTec lists the dataset under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International license and prohibits commercial dataset use.

License information:  
https://creativecommons.org/licenses/by-nc-sa/4.0/

Dataset images and masks are not included. This project is an educational baseline, not a production-ready inspection system.
