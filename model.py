import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum


class MambaBlock(nn.Module):
    def __init__(self, dim, heads):
        """
        Mamba Block: Multi-head self-attention block.

        Args:
            dim (int): Dimension of the input tensor.
            heads (int): Number of attention heads.
        """
        super().__init__()
        self.heads = heads
        self.scale = dim**-0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(0.1))

    def forward(self, x, mask=None):
        """
        Forward pass of the Mamba Block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            mask (torch.Tensor, optional): Attention mask of shape (batch_size, seq_len). Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim).
        """
        # Split the input tensor into query, key, and value tensors
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        # Compute the attention scores
        dots = einsum("bhid,bhjd->bhij", q, k) * self.scale

        # Apply the attention mask if provided
        if mask is not None:
            mask = F.pad(
                mask.flatten(1), (0, dots.size(-1) - mask.size(-1)), value=True
            )
            dots = dots.masked_fill(~mask[:, None, None, :], float("-inf"))

        # Compute the attention probabilities
        attn = F.softmax(dots, dim=-1)

        # Apply the attention to the value tensor
        out = einsum("bhij,bhjd->bhid", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        # Apply the output projection
        return self.to_out(out)


class EinFFTChannelMixer(nn.Module):
    def __init__(self, channels):
        """
        EinFFT Channel Mixer: Mixes channels using Fourier Transform.

        Args:
            channels (int): Number of channels in the input tensor.
        """
        super().__init__()
        self.channels = channels

        self.weight_real = nn.Parameter(torch.randn(channels // 2 + 1))
        self.weight_imag = nn.Parameter(torch.randn(channels // 2 + 1))

    def forward(self, x):
        """
        Forward pass of the EinFFT Channel Mixer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, channels).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, channels).
        """
        # Compute the Fourier Transform of the input tensor
        x_fft = torch.fft.rfft(x, dim=1)

        # Apply the mixing weights to the real and imaginary parts of the Fourier coefficients
        x_fft_real = einsum("...c,c->...c", x_fft.real, self.weight_real)
        x_fft_imag = einsum("...c,c->...c", x_fft.imag, self.weight_imag)

        # Compute the inverse Fourier Transform to obtain the mixed tensor
        x_ifft_real = torch.fft.irfft(torch.complex(x_fft_real, x_fft_imag), dim=1)
        return x_ifft_real


class SiMBA(nn.Module):
    def __init__(self, image_size, channels, num_blocks, heads):
        """
        SiMBA: Simple Masked Block Attention.

        Args:
            image_size (int): Size of the input image.
            channels (int): Number of channels in the input tensor.
            num_blocks (int): Number of Mamba Blocks.
            heads (int): Number of attention heads.
        """
        super().__init__()
        self.channels = channels
        self.image_size = image_size

        self.initial_conv = nn.Conv2d(3, channels, kernel_size=7, stride=2, padding=3)

        self.mamba_blocks = nn.ModuleList(
            [MambaBlock(dim=channels, heads=heads) for _ in range(num_blocks)]
        )

        self.einfft_mixer = EinFFTChannelMixer(channels=channels)

        self.head = nn.Linear(channels * image_size // (2 ** (num_blocks + 1)), 1000)

    def forward(self, x):
        """
        Forward pass of the SiMBA model.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        # Apply the initial convolution
        x = self.initial_conv(x)
        x = rearrange(x, "b c h w -> b (h w) c")

        # Apply the Mamba Blocks
        for mamba_block in self.mamba_blocks:
            x = mamba_block(x)

        # Apply the EinFFT Channel Mixer
        x = self.einfft_mixer(x)

        # Flatten the tensor and apply the final linear layer
        x = x.view(x.size(0), -1)
        return self.head(x)
