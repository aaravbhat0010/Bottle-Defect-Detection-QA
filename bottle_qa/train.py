from pathlib import Path
import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from bottle_qa.data import load_image
from bottle_qa.model import BottleAutoencoder
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
args = parser.parse_args()
data_root = args.root


import random

# Keep the split the same each time.
SEED = 42

# Find all normal bottle photos.
normal_paths = sorted(
    (data_root / "train" / "good").glob("*.png")
)

assert len(normal_paths) >= 2, "Need at least two normal images."

# Copy the list and mix up the order.
shuffled_paths = normal_paths.copy()
random.Random(SEED).shuffle(shuffled_paths)

# Save 20% for validation.
validation_count = max(1, round(len(shuffled_paths) * 0.2))

val_paths = shuffled_paths[:validation_count]
train_paths = shuffled_paths[validation_count:]

# Check that no file is in both groups.
assert set(train_paths).isdisjoint(val_paths)

# Check that we included every file.
assert len(train_paths) + len(val_paths) == len(normal_paths)

print("Training images:", len(train_paths))
print("Validation images:", len(val_paths))
print("Split checks passed")

class BottleDataset(Dataset):  # Defines our collection of bottle photos.
    def __init__(self, paths):  # Runs when we create a dataset.
        self.paths = list(paths)  # Stores the photo locations.

    def __len__(self):  # Tells PyTorch how many photos we have.
        return len(self.paths)  # Returns the photo count.

    def __getitem__(self, index):  # Tells PyTorch how to get one photo.
        return load_image(self.paths[index])


train_dataset = BottleDataset(train_paths)  # Creates the training collection.
val_dataset = BottleDataset(val_paths)  # Creates the validation collection.

generator = torch.Generator().manual_seed(SEED)  # Makes training shuffles repeatable.

train_loader = DataLoader(
    train_dataset,
    batch_size=16,  # Gives the model up to 16 photos at a time.
    shuffle=True,  # Mixes the training order each time through the dataset.
    generator=generator
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,  # Also checks validation photos in groups of 16.
    shuffle=False  # Keeps validation order fixed.
)



import copy

# Start with repeatable random weights and photo order.
torch.manual_seed(SEED)
generator.manual_seed(SEED)

model = BottleAutoencoder()

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 20
best_val_loss = float("inf")
best_weights = None
history = []

for epoch in range(EPOCHS):
    # Learn from the 167 training photos.
    model.train()
    training_total = 0.0

    for images in train_loader:
        optimizer.zero_grad()

        recreated = model(images)
        loss = loss_function(recreated, images)

        assert torch.isfinite(loss), "Training error is invalid."

        loss.backward()
        optimizer.step()

        training_total += loss.item() * images.size(0)

    train_loss = training_total / len(train_dataset)

    # Check the 42 validation photos without learning from them.
    model.eval()
    validation_total = 0.0

    with torch.no_grad():
        for images in val_loader:
            recreated = model(images)
            loss = loss_function(recreated, images)

            assert torch.isfinite(loss), "Validation error is invalid."

            validation_total += loss.item() * images.size(0)

    val_loss = validation_total / len(val_dataset)

    history.append({
        "train_loss": train_loss,
        "val_loss": val_loss
    })

    # Remember the model with the lowest validation error.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights = copy.deepcopy(model.state_dict())

    print(
        f"Round {epoch + 1}/{EPOCHS} | "
        f"Training error: {train_loss:.5f} | "
        f"Validation error: {val_loss:.5f}",
        flush=True
    )

# Restore and save the best model from this run.
model.load_state_dict(best_weights)
model.eval()

Path("artifacts").mkdir(exist_ok=True)

torch.save({
    "model_state_dict": model.state_dict(),
    "seed": SEED,
    "image_size": 128,
    "epochs": EPOCHS,
    "best_val_loss": best_val_loss,
    "history": history,
    "train_paths": [str(path) for path in train_paths],
    "val_paths": [str(path) for path in val_paths]
}, "artifacts/bottle_autoencoder.pt")

print("\nTraining finished.")
print("Best validation error:", best_val_loss)
print("Saved to artifacts/bottle_autoencoder.pt")

model.eval()  # Checks photos without updating the model.
validation_scores = []  # Stores one reconstruction error per photo.

with torch.no_grad():  # Turns off gradient calculations.
    for images in val_loader:  # Goes through all 42 normal validation photos.
        recreated = model(images)  # Recreates each photo.

        scores = ((recreated - images) ** 2).mean(dim=(1, 2, 3))  # Finds each photo's error.
        validation_scores.extend(scores.tolist())  # Adds those errors to our list.

validation_scores = np.array(validation_scores)  # Converts the list to a NumPy array.

assert len(validation_scores) == len(val_dataset)  # Checks we scored all validation photos.
assert np.isfinite(validation_scores).all()  # Checks the scores are valid.

threshold = float(np.quantile(validation_scores, 0.95))  # Sets our cutoff.

print("Validation photos checked:", len(validation_scores))
print("Defect cutoff:", threshold)
print("Normal photos above cutoff:", int((validation_scores > threshold).sum()))

from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix
)  # Loads tools for measuring predictions.

test_paths = sorted((data_root / "test").glob("*/*.png"))  # Finds the real test photos.
assert test_paths, "No test photos found."

test_labels = np.array([
    0 if path.parent.name == "good" else 1
    for path in test_paths
])  # Records the known answer: 0 = normal, 1 = defective.

test_dataset = BottleDataset(test_paths)  # Uses the same image preparation as training.
test_loader = DataLoader(
    test_dataset, batch_size=16, shuffle=False
)  # Keeps photos in the same order as their labels.

test_scores = []  # Stores one reconstruction error per test photo.
model.eval()  # Puts the model in evaluation mode.

with torch.no_grad():  # Checks photos without learning from them.
    for images in test_loader:
        recreated = model(images)  # Recreates the test photos.
        scores = ((recreated - images) ** 2).mean(dim=(1, 2, 3))  # Measures each photo's error.
        test_scores.extend(scores.tolist())  # Saves those errors.

test_scores = np.array(test_scores)  # Converts scores into an array.
assert len(test_scores) == len(test_labels)  # Checks every photo has a score.
assert np.isfinite(test_scores).all()  # Checks all scores are valid.
assert set(test_labels.tolist()) == {0, 1}  # Checks both normal and defective photos exist.

test_predictions = (test_scores > threshold).astype(int)  # Flags photos above our fixed cutoff.

print("Normal test photos:", int((test_labels == 0).sum()))
print("Defective test photos:", int((test_labels == 1).sum()))
print("ROC-AUC:", roc_auc_score(test_labels, test_scores))
print("Precision:", precision_score(test_labels, test_predictions, zero_division=0))
print("Recall:", recall_score(test_labels, test_predictions, zero_division=0))
print("F1:", f1_score(test_labels, test_predictions, zero_division=0))

matrix = confusion_matrix(test_labels, test_predictions, labels=[0, 1])
print("\nConfusion matrix:")
print(matrix)
print("Rows = actual [normal, defective]")
print("Columns = predicted [normal, defective]")

import json
Path("artifacts/threshold.json").write_text(json.dumps({"threshold": threshold}))
