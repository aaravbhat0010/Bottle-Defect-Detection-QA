import torch
from bottle_qa.model import BottleAutoencoder


def test_model_output():
    model = BottleAutoencoder()  # Creates an untrained model.
    model.eval()  # Sets evaluation mode.

    images = torch.rand(2, 3, 128, 128)  # Creates two fake images.

    with torch.no_grad():  # Runs without learning.
        recreated = model(images)  # Compresses and recreates them.

    assert recreated.shape == images.shape  # Checks the output dimensions.
    assert torch.isfinite(recreated).all()  # Checks all numbers are valid.
    assert recreated.min() >= 0  # Checks values stay above or equal to zero.
    assert recreated.max() <= 1  # Checks values stay below or equal to one.