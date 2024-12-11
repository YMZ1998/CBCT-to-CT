import numpy as np
import torch


def to_numpy_squeeze(input_data):
    """Convert PyTorch tensor or NumPy array to NumPy array, and squeeze the dimensions."""
    if isinstance(input_data, np.ndarray):
        # If input is already a NumPy array, squeeze and return
        return np.squeeze(input_data)
    elif isinstance(input_data, torch.Tensor):
        # If input is a PyTorch tensor, detach and convert to NumPy, then squeeze
        return input_data.detach().cpu().squeeze().numpy() if input_data.requires_grad else input_data.cpu().squeeze().numpy()
    else:
        raise TypeError(f"Expected input to be a PyTorch tensor or a NumPy array, but got {type(input_data)}.")


def post_process(out, ct, location, mask=None, min_v=-1024, max_v=3000):
    out = to_numpy_squeeze(out)
    ct = to_numpy_squeeze(ct) if ct is not None else None
    location = to_numpy_squeeze(location)
    mask = to_numpy_squeeze(mask) if mask is not None else None

    # Normalize to [0, 1] range
    out = (out + 1) / 2
    ct = (ct + 1) / 2 if ct is not None else None

    # Rescale to the specified range
    out = out * (max_v - min_v) + min_v
    ct = ct * (max_v - min_v) + min_v if ct is not None else None

    # Clip values within the specified range
    out = np.clip(out, min_v, max_v)
    ct = np.clip(ct, min_v, max_v) if ct is not None else None

    # Calculate the cropping boundaries
    y_min, x_min = int(location[1]), int(location[0])
    y_max, x_max = y_min + int(location[3]), x_min + int(location[2])

    # Crop the images
    out = out[y_min:y_max, x_min:x_max]
    ct = ct[y_min:y_max, x_min:x_max] if ct is not None else None
    mask = mask[y_min:y_max, x_min:x_max] if mask is not None else None

    return out, ct, mask
