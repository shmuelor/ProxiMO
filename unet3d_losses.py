"""Implementation of Gradient Difference Loss and Total Variation Loss for 3D Images"""

import torch
import torch.nn.functional as F


def gradient_difference_loss(
    inputs: torch.Tensor, targets: torch.Tensor, alpha: float = 1.0, mode: int = 1
):
    """
    Compute gradient difference loss for 3D images.
    Refs:
        - https://arxiv.org/pdf/1511.05440
        - https://pubmed.ncbi.nlm.nih.gov/29894829/

    Args:
        inputs (torch.Tensor): The predicted image, shape (batch_size, channels, depth, height, width).
        targets (torch.Tensor): The ground truth image, shape (batch_size, channels, depth, height, width).
        alpha (float): see ref [1] Eq. 6.
        mode (int): choose from 1 or 2.

    Returns:
        torch.Tensor: The gradient difference loss. scalar.
    """
    inputs_dz, inputs_dy, inputs_dx = compute_image_gradients(inputs)
    targets_dz, targets_dy, targets_dx = compute_image_gradients(targets)

    if mode == 1:
        inputs_dz = inputs_dz.abs()
        inputs_dy = inputs_dy.abs()
        inputs_dx = inputs_dx.abs()
        targets_dz = targets_dz.abs()
        targets_dy = targets_dy.abs()
        targets_dx = targets_dx.abs()

    loss = torch.mean(
        (inputs_dz - targets_dz).abs().pow(alpha)
        + (inputs_dy - targets_dy).abs().pow(alpha)
        + (inputs_dx - targets_dx).abs().pow(alpha)
    )

    return loss


def compute_image_gradients(img: torch.Tensor):
    """Compute image gradients for 3D images.

    Args:
        img (torch.Tensor): shape (batch_size, channels, depth, height, width).

    Returns:
        tuple: (grad_z, grad_y, grad_x), each with shape (batch_size, channels, depth, height, width).
    """
    dx = img[:, :, :, :, 1:] - img[:, :, :, :, :-1]
    dy = img[:, :, :, 1:, :] - img[:, :, :, :-1, :]
    dz = img[:, :, 1:, :, :] - img[:, :, :-1, :, :]

    dx = F.pad(dx, (0, 1), mode="constant", value=0)
    dy = F.pad(dy, (0, 0, 0, 1), mode="constant", value=0)
    dz = F.pad(dz, (0, 0, 0, 0, 0, 1), mode="constant", value=0)

    return dz, dy, dx


def total_variation_loss(inputs: torch.Tensor):
    """
    Compute total variation loss for 3D images.
    Refs:
        - https://arxiv.org/abs/1412.0035
        - https://arxiv.org/pdf/2008.06187

    Args:
        inputs (torch.Tensor): The predicted image, shape (batch_size, channels, depth, height, width).

    Returns:
        torch.Tensor: The total variation loss. scalar.
    """
    dz, dy, dx = compute_image_gradients(inputs)
    return torch.mean(dz.abs() + dy.abs() + dx.abs())
