
import math
import torch
from .data import load_image
from .model import BottleAutoencoder


def label_from_path(path):
    # Read the real answer from the dataset folder name.
    return 0 if path.parent.name == "good" else 1


def predict_image(image_path, checkpoint_path, threshold):
    # Reject an invalid cutoff.
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("Invalid threshold")

    # Open saved model weights on CPU.
    saved = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    model = BottleAutoencoder()
    model.load_state_dict(saved["model_state_dict"])
    model.eval()

    # Open the photo and put it into a batch of one.
    image = load_image(image_path).unsqueeze(0)

    with torch.no_grad():
        recreated = model(image)  # Recreate the photo.
        score = ((recreated - image) ** 2).mean().item()  # Measure its error.

    if not math.isfinite(score):
        raise ValueError("Invalid score")

    # Flag the photo if its error exceeds the cutoff.
    return {"score": score, "is_defective": score > threshold}
