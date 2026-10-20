"""Innovation 1: Empirical-background masking strategy."""

from __future__ import annotations
import numpy as np
import torch

from data.preprocessing import TabularPreprocessor


def apply_background_mask(
    inputs: torch.Tensor,
    feature_mask: torch.Tensor,
    preprocessor: TabularPreprocessor,
    background_bank: np.ndarray,
    rng: np.random.Generator,
    K: int = 1,
) -> torch.Tensor:
    """Replace masked features with real values from background bank.

    For each sample in the batch:
      - features where mask==1: keep the original input
      - features where mask==0: replace with values from a randomly
        sampled background row, preserving one-hot group validity

    When K > 1, returns K copies stacked for averaging of targets.
    Output shape: (batch_size * K, input_dim)
    """
    batch_size = inputs.shape[0]
    device = inputs.device
    input_dim = inputs.shape[1]

    # Sample background indices: (batch_size, K)
    bg_indices = rng.integers(0, len(background_bank), size=(batch_size, K))

    all_masked: list[torch.Tensor] = []
    for k in range(K):
        bg_rows = torch.from_numpy(
            background_bank[bg_indices[:, k]].astype(np.float32)
        ).to(device)

        # Build per-column mask by expanding feature-level mask
        masked_input = inputs.clone()
        for fi, fn in enumerate(preprocessor.feature_order):
            group = preprocessor.group(fn)
            col_range = slice(group.start, group.end)
            # Where mask is 0, replace with background
            mask_col = feature_mask[:, fi:fi+1]  # (batch, 1)
            masked_input[:, col_range] = (
                mask_col * inputs[:, col_range] +
                (1.0 - mask_col) * bg_rows[:, col_range]
            )
        all_masked.append(masked_input)

    # Stack: (batch_size * K, input_dim)
    return torch.cat(all_masked, dim=0)


def compute_background_targets(
    blackbox_model: object,
    inputs: torch.Tensor,
    feature_mask: torch.Tensor,
    preprocessor: TabularPreprocessor,
    background_bank: np.ndarray,
    rng: np.random.Generator,
    device: torch.device,
    K: int = 1,
) -> torch.Tensor:
    """Compute averaged black-box outputs over K background replacements.

    Returns: (batch_size, output_dim) averaged targets.
    """
    from training.evaluate import predict_raw_outputs

    masked_inputs = apply_background_mask(
        inputs, feature_mask, preprocessor, background_bank, rng, K
    )  # (batch * K, dim)

    # Get blackbox outputs for all masked versions
    raw = predict_raw_outputs(
        blackbox_model, masked_inputs.detach().cpu().numpy(), device
    )  # (batch * K, output_dim)

    raw_tensor = torch.from_numpy(raw.astype(np.float32)).to(device)
    batch_size = inputs.shape[0]

    if K == 1:
        return raw_tensor

    # Average over K background samples
    output_dim = raw_tensor.shape[1]
    reshaped = raw_tensor.view(K, batch_size, output_dim)
    return reshaped.mean(dim=0)
