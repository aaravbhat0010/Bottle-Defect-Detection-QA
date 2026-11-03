
from pathlib import Path
import pytest
import torch
from PIL import Image
from bottle_qa.model import BottleAutoencoder
from bottle_qa.inference import label_from_path, predict_image


def test_learning_changes_weights():
    torch.manual_seed(42)  # Makes the test repeatable.
    model = BottleAutoencoder()  # Creates a separate model.
    images = torch.rand(2, 3, 128, 128)  # Creates two fake images.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    before = [p.detach().clone() for p in model.parameters()]  # Copies starting weights.

    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(model(images), images)
    assert torch.isfinite(loss)  # Checks the error is valid.

    loss.backward()  # Calculates weight adjustments.

    for parameter in model.parameters():
        assert parameter.grad is not None  # Checks gradients exist.
        assert torch.isfinite(parameter.grad).all()  # Checks gradients are valid.

    optimizer.step()  # Changes the weights.

    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, model.parameters())
    )  # Checks that at least one weight tensor changed.


def test_saving_keeps_predictions(tmp_path):
    torch.manual_seed(42)
    model = BottleAutoencoder().eval()
    images = torch.rand(2, 3, 128, 128)

    with torch.no_grad():
        before = model(images)  # Gets predictions before saving.

    checkpoint = tmp_path / "model.pt"  # Uses a temporary file.
    torch.save({"model_state_dict": model.state_dict()}, checkpoint)

    restored = BottleAutoencoder().eval()  # Creates another model.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restored.load_state_dict(saved["model_state_dict"])  # Restores saved weights.

    with torch.no_grad():
        after = restored(images)  # Gets predictions after reopening.

    torch.testing.assert_close(before, after, rtol=0, atol=0)  # Checks they match.


@pytest.mark.parametrize("folder, expected", [
    ("good", 0),
    ("broken_large", 1),
    ("broken_small", 1),
    ("contamination", 1),
])
def test_labels(folder, expected):
    path = Path("test") / folder / "001.png"
    assert label_from_path(path) == expected  # Checks normal/defective labels.


def test_complete_prediction(tmp_path):
    image_path = tmp_path / "red.png"
    Image.new("RGB", (20, 30), (255, 0, 0)).save(image_path)  # Makes a fake photo.

    model = BottleAutoencoder().eval()
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint)

    result = predict_image(image_path, checkpoint, threshold=1.0)  # Runs the whole workflow.

    assert 0 < result["score"] <= 1  # Checks the error score.
    assert result["is_defective"] is False  # Score cannot exceed this cutoff.

    lower_cutoff = result["score"] / 2  # Sets a cutoff below the same score.
    flagged = predict_image(image_path, checkpoint, lower_cutoff)
    assert flagged["is_defective"] is True  # Checks that flagging works too.


def test_bad_cutoff_rejected(tmp_path):
    with pytest.raises(ValueError, match="threshold"):
        predict_image(
            tmp_path / "unused.png",
            tmp_path / "unused.pt",
            threshold=float("nan"),
        )  # Checks an invalid cutoff is rejected before opening files.
