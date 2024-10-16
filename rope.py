import torch

def rope_cos_sin(head_dim, theta_base = 10000, context_length = 4096, freq_config = None):
    assert head_dim % 2 == 0, "Embedding dim not even"

    # Compute inverse frequency 
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim // 2) / (head_dim // 2)))

    # Adjust frequency 
    if freq_config is not None:
        low_freq_wavelen = freq_config["original_context_length"] / freq_config["low_freq_factor"]
        high_freq_wavelen = freq_config["original_context_length"] / freq_config["high_freq_factor"]

        wavelen = 2 * torch.pi / inv_freq # convert frequency back to wavelength

        inv_freq_llama = torch.where(
            wavelen > low_freq_wavelen, inv_freq / freq_config["factor"], inv_freq # 
        )

        smooth_factor = (freq_config["original_context_length"] / wavelen - freq_config["low_freq_factor"]) / (
            freq_config["high_freq_factor"] - freq_config["low_freq_factor"]
        )

        smoothed_inv_freq = (
            (1 - smooth_factor) * (inv_freq / freq_config["factor"]) + smooth_factor * inv_freq
        )

        is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        inv_freq = inv_freq_llama

        # Position indices
        positions = torch.arange(context_length) 

        # Compute angles
        angles = positions[:, None] * inv_freq[None, :] 

        # Shape: (context_length, head_dim // 2) -> (context_length, head_dim) 
        angles = torch.cat([angles, angles], dim = 1) 

        cos, sin = torch.cos(angles), torch.sin(angles)

        return cos, sin
    
def compute_rope(x, cos, sin):
    batch_size, num_heads, seq_len, head_dim = x.shape

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  
    x2 = x[..., head_dim // 2 :]  

    # Adjust sin and cos shapes
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)
