
import torch
from PIL import Image
from bottle_qa.data import load_image


def test_red_image_preprocessing(tmp_path):
    path = tmp_path / "red.png"  # Uses a temporary folder.
    Image.new("RGB", (20, 30), (255, 0, 0)).save(path)  # Creates a red image.

    result = load_image(path)  # Runs our actual preprocessing code.

    assert result.shape == (3, 128, 128)  # Checks channels, height and width.
    assert result.dtype == torch.float32  # Checks the number format.
    assert torch.isfinite(result).all()  # Checks for invalid numbers.
    assert torch.all(result[0] == 1)  # Red should become 255 / 255 = 1.
    assert torch.all(result[1:] == 0)  # Green and blue should stay zero.


def test_corrupt_image_rejected(tmp_path):
    import pytest  # Lets us check that an expected error occurs.
    from PIL import UnidentifiedImageError

    path = tmp_path / "broken.png"  # Creates a temporary file location.
    path.write_bytes(b"this is not an image")  # Writes invalid image contents.

    with pytest.raises(UnidentifiedImageError):  # Expects the loader to reject it.
        load_image(path)  # Tries opening the corrupt file.
