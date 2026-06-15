import torch
import torch.nn.functional as F


def pad_to_window_size(
    image_features: torch.Tensor,
    window_size: int,
) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, but received {window_size}.")
    if image_features.ndim != 4:
        raise ValueError(
            "Expected image features with shape [B, C, H, W], "
            f"but received {tuple(image_features.shape)}."
        )

    _, _, height, width = image_features.shape
    pad_height = -height % window_size
    pad_width = -width % window_size
    if pad_height or pad_width:
        image_features = F.pad(image_features, (0, pad_width, 0, pad_height))

    padded_height, padded_width = image_features.shape[-2:]
    return image_features, (height, width), (padded_height, padded_width)


def partition_padded_windows(
    padded_features: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, but received {window_size}.")
    if padded_features.ndim != 4:
        raise ValueError(
            "Expected padded features with shape [B, C, H, W], "
            f"but received {tuple(padded_features.shape)}."
        )

    batch_size, channels, padded_height, padded_width = padded_features.shape
    if padded_height % window_size != 0 or padded_width % window_size != 0:
        raise ValueError(
            "Padded features must have spatial dimensions divisible by window_size, "
            f"but received {(padded_height, padded_width)} with {window_size=}."
        )

    windows = padded_features.reshape(
        batch_size,
        channels,
        padded_height // window_size,
        window_size,
        padded_width // window_size,
        window_size,
    )
    windows = windows.permute(0, 2, 4, 3, 5, 1).reshape(
        -1,
        window_size * window_size,
        channels,
    )
    return windows


def merge_padded_windows(
    window_tokens: torch.Tensor,
    padded_shape: tuple[int, int],
    window_size: int,
) -> torch.Tensor:
    if window_tokens.ndim != 3:
        raise ValueError(
            "Expected window tokens with shape [B*num_windows, window_size^2, C], "
            f"but received {tuple(window_tokens.shape)}."
        )

    padded_height, padded_width = padded_shape
    windows_per_image = (padded_height // window_size) * (padded_width // window_size)

    if window_tokens.shape[0] % windows_per_image != 0:
        raise ValueError(
            "Window batch dimension does not align with the padded image size: "
            f"{window_tokens.shape[0]=}, {windows_per_image=}."
        )

    batch_size = window_tokens.shape[0] // windows_per_image
    channels = window_tokens.shape[-1]

    image_grid = window_tokens.reshape(
        batch_size,
        padded_height // window_size,
        padded_width // window_size,
        window_size,
        window_size,
        channels,
    )
    image_grid = image_grid.permute(0, 5, 1, 3, 2, 4).reshape(
        batch_size,
        channels,
        padded_height,
        padded_width,
    )
    return image_grid


def build_shifted_window_attention_mask(
    padded_shape: tuple[int, int],
    window_size: int,
    shift_size: int,
    *,
    device: torch.device,
) -> torch.Tensor | None:
    if shift_size == 0:
        return None
    if not 0 <= shift_size < window_size:
        raise ValueError(
            f"shift_size must satisfy 0 <= shift_size < window_size, got {shift_size=} and "
            f"{window_size=}."
        )

    padded_height, padded_width = padded_shape
    image_mask = torch.zeros((1, padded_height, padded_width, 1), device=device, dtype=torch.float32)
    height_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    width_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )

    mask_index = 0
    for height_slice in height_slices:
        for width_slice in width_slices:
            image_mask[:, height_slice, width_slice, :] = mask_index
            mask_index += 1

    mask_windows = partition_padded_windows(image_mask.permute(0, 3, 1, 2), window_size)
    mask_windows = mask_windows.squeeze(-1)
    return mask_windows.unsqueeze(1) != mask_windows.unsqueeze(2)


def build_padding_attention_mask(
    original_shape: tuple[int, int],
    padded_shape: tuple[int, int],
    window_size: int,
    shift_size: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    original_height, original_width = original_shape
    padded_height, padded_width = padded_shape

    valid = torch.zeros((1, 1, padded_height, padded_width), device=device, dtype=torch.bool)
    valid[..., :original_height, :original_width] = True

    if shift_size > 0:
        valid = torch.roll(
            valid,
            shifts=(-shift_size, -shift_size),
            dims=(-2, -1),
        )

    valid_windows = partition_padded_windows(valid.float(), window_size).squeeze(-1).bool()
    # [num_windows, L]

    invalid_keys = ~valid_windows.unsqueeze(1)
    # [num_windows, 1, L]

    invalid_queries = ~valid_windows.unsqueeze(2)
    # [num_windows, L, 1]

    return invalid_queries | invalid_keys
