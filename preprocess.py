from PIL import Image
import numpy as np
import torch


def preprocess_image(
    img: Image.Image,
    image_size: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    
    img = img.convert("RGB")
    img = img.resize((image_size, image_size), Image.BICUBIC)  # type: ignore

    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3), [0, 1]
    arr = (arr - 0.5) / 0.5  # [-1, 1]

    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
    tensor = tensor.unsqueeze(0).to(dtype)  # (1, 3, H, W)
    return tensor
