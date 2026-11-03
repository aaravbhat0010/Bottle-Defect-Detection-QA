from PIL import Image
import numpy as np
import torch

def load_image(path):  # Creates a reusable function that takes a photo's location.
    with Image.open(path) as image:  # Opens the photo and closes the file automatically afterward.
        image = image.convert("RGB")  # Gives the photo three color channels: red, green, and blue.
        image = image.resize((128, 128), Image.Resampling.BILINEAR)  # Resizes the photo to 128 pixels wide and tall.
        pixels = np.array(image, dtype=np.float32) / 255.0  # Converts the photo into decimal numbers between 0 and 1.

    return torch.from_numpy(pixels).permute(2, 0, 1).contiguous()
