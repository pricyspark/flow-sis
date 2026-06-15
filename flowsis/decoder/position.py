import torch

def build_2d_sincos_pos_encoding(
    height: int,
    width: int,
    channels: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if channels % 4 != 0:
        raise ValueError(
            "2D sinusoidal positional encoding requires channels divisible by 4, "
            f"but received channels={channels}."
        )
        
    if height <= 0 or width <= 0:
        raise ValueError(f"height and width must be positive, got {height=} and {width=}.")
    if channels <= 0:
        raise ValueError(f"channels must be positive, got {channels=}.")

    quarter_channels = channels // 4
    frequencies = torch.arange(
        quarter_channels, 
        device=device, 
        dtype=torch.float32
    )   # (C/4)
    frequencies = 1.0 / (10000 ** (frequencies / quarter_channels))             # (C/4,)
    
    y_coords = torch.arange(height, device=device, dtype=torch.float32)         # (H,)
    x_coords = torch.arange(width, device=device, dtype=torch.float32)          # (W,)

    y_angles = y_coords[:, None] * frequencies[None, :]                         # (H,C/4)
    x_angles = x_coords[:, None] * frequencies[None, :]                         # (W,C/4)

    y_encoding = torch.cat([y_angles.sin(), y_angles.cos()], dim=1)             # (H,C/2)
    x_encoding = torch.cat([x_angles.sin(), x_angles.cos()], dim=1)             # (W,C/2)

    y_encoding = y_encoding[:, None, :].expand(height, width, channels // 2)    # (H,W,C/2)
    x_encoding = x_encoding[None, :, :].expand(height, width, channels // 2)    # (H,W,C/2)
    
    positional_encoding = torch.cat([y_encoding, x_encoding], dim=-1)           # (H,W,C)
    positional_encoding = positional_encoding.reshape(1, height * width, channels)  # (1,H*W,C)
    
    return positional_encoding.to(dtype=dtype)  # (1,H*W,C)


def build_reference_grid(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if height <= 0 or width <= 0:
        raise ValueError(f"height and width must be positive, got {height=} and {width=}.")

    y_coords = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    x_coords = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)