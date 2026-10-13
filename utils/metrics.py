import numpy as np
from skimage.metrics import structural_similarity as ssim


def compute_ssim(img1, img2):
    """RGB SSIM. Returns Python float."""
    return float(ssim(img1, img2, channel_axis=2, data_range=255))


def compute_psnr(img1, img2):
    """Returns Python float."""
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def compute_lpips(img1, img2, device="cpu"):
    """Returns Python float or None."""
    try:
        import torch
        import lpips
        loss_fn = lpips.LPIPS(net="alex").to(device)
        t1 = torch.from_numpy(img1).permute(2, 0, 1).float() / 127.5 - 1
        t2 = torch.from_numpy(img2).permute(2, 0, 1).float() / 127.5 - 1
        with torch.no_grad():
            d = loss_fn(t1.unsqueeze(0).to(device), t2.unsqueeze(0).to(device))
        return float(d.item())
    except ImportError:
        return None