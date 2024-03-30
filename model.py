from typing import Optional

import torch
from einops.layers.torch import Rearrange, Reduce
from torch import nn, Tensor
from zeta import MambaBlock


def posemb_sincos_2d(
        h: int,
        w: int,
        dim: int,
        temperature: int = 10000,
        dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Generate 2D sinusoidal positional embeddings.

    Args:
        h (int): Height of the image.
        w (int): Width of the image.
        dim (int): Dimension of the positional embeddings.
        temperature (int, optional): Temperature for the sinusoidal function. Defaults to 10000.
        dtype (torch.dtype, optional): Data type of the positional embeddings. Defaults to torch.float32.

    Returns:
        Tensor: 2D sinusoidal positional embeddings of shape (h*w, dim).
    """
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (
                   dim % 4
           ) == 0, "Embedding dimension must be divisible by 4 for sincos embeddings"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class EinFFTText(nn.Module):
    """
    EinFFTText module performs an FFT-based operation on the input tensor, with options for dynamic activation functions,
    dropout regularization, and multi-scale processing.

    Args:
        sequence_length (int): Length of the input sequence.
        dim (int): Dimension of the model.
        heads (int, optional): Number of attention heads. Defaults to 1.
        activation_fn (Callable[[Tensor], Tensor], optional): Activation function. Defaults to nn.SiLU().
        dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
        multi_scale (bool, optional): Enables multi-scale processing if True. Defaults to False.
        scales (list, optional): List of scales for multi-scale processing. Defaults to [1.0, 0.5, 0.25].
        sparsity_threshold (float, optional): Threshold for soft thresholding. Defaults to 0.01.

    Attributes:
        sequence_length (int): Length of the input sequence.
        dim (int): Dimension of the model.
        heads (int): Number of attention heads.
        multi_scale (bool): Whether to use multi-scale processing.
        scales (list): List of scales for multi-scale processing.
        sparsity_threshold (float): Threshold for soft thresholding.
        activation (Callable[[Tensor], Tensor]): Activation function.
        dropout (nn.Dropout): Dropout module.
        complex_weight (nn.Parameter): Learnable complex weight parameter.
        real_weight (nn.Parameter): Learnable real weight parameter.
    """

    def __init__(
            self,
            sequence_length: int,
            dim: int,
            heads: int = 1,
            activation_fn: nn.Module = nn.SiLU(),
            dropout_rate: float = 0.1,
            multi_scale: bool = False,
            scales: Optional[list] = None,
            sparsity_threshold: float = 0.01,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.dim = dim
        self.heads = heads
        self.multi_scale = multi_scale
        self.scales = scales if scales is not None else [1.0, 0.5, 0.25]
        self.sparsity_threshold = sparsity_threshold

        self.activation = activation_fn
        self.dropout = nn.Dropout(p=dropout_rate)

        self.complex_weight = nn.Parameter(
            torch.randn(heads, sequence_length, dim // heads, dtype=torch.cfloat)
            / sequence_length
        )
        self.real_weight = nn.Parameter(
            torch.randn(heads, sequence_length, dim // heads) / sequence_length
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the EinFFTText module.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, sequence_length, dim).
        """
        if self.multi_scale:
            x = self._process_multi_scale(x)

        batch_size, _, _ = x.shape

        # Apply 1D FFT to transform input tensor to frequency domain
        freq_domain = torch.fft.fft(x, dim=-1, norm="ortho")

        # Split real and imaginary parts
        real_freq, imag_freq = freq_domain.real, freq_domain.imag

        # Perform complex-valued multiplication using Einstein summation
        complex_mul_real = torch.einsum(
            "bsd,hsf->bhsf", real_freq, self.complex_weight.real
        )
        complex_mul_imag = torch.einsum(
            "bsd,hsf->bhsf", imag_freq, self.complex_weight.imag
        )
        complex_mul = complex_mul_real - complex_mul_imag

        # Apply activation and dropout to real and imaginary parts separately
        activated_real = self.dropout(self.activation(complex_mul.real))
        activated_imag = self.dropout(self.activation(complex_mul.imag))

        # Element-wise multiplication with real and complex weights using Einstein summation
        emm_real = torch.einsum("bhsf,hsf->bsd", activated_real, self.real_weight)
        emm_imag = torch.einsum(
            "bhsf,hsf->bsd", activated_imag, self.complex_weight.real
        )

        # Combine real and imaginary parts
        emm_complex = torch.complex(emm_real, emm_imag)

        # Apply soft thresholding for sparsity
        emm_complex = self._soft_threshold(emm_complex, self.sparsity_threshold)

        # Apply 1D IFFT to transform back to time domain
        output = torch.fft.ifft(emm_complex, dim=-1, norm="ortho").real

        return output

    def _process_multi_scale(self, x: Tensor) -> Tensor:
        """
        Processes the input tensor at multiple scales and combines the results.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, dim).

        Returns:
            Tensor: Processed tensor after multiscale processing.
        """
        multi_scale_features = []

        for scale in self.scales:
            if scale == 1.0:
                # No need to rescale if scale is 1.0
                multi_scale_features.append(x)
            else:
                # Rescale the input tensor to the current scale
                scaled_seq_length = int(self.sequence_length * scale)
                scaled_x = nn.functional.interpolate(
                    x, size=(scaled_seq_length, self.dim), mode="nearest"
                )
                # Process the scaled input tensor using EinFFT
                processed_scaled_x = self.forward(scaled_x)
                # Upsample the processed tensor back to the original sequence length
                upsampled_x = nn.functional.interpolate(
                    processed_scaled_x,
                    size=(self.sequence_length, self.dim),
                    mode="nearest",
                )
                multi_scale_features.append(upsampled_x)

        # Concatenate the multiscale features along the channel dimension
        multi_scale_output = torch.cat(multi_scale_features, dim=-1)

        return multi_scale_output

    @classmethod
    def _soft_threshold(cls, x: Tensor, threshold: float) -> Tensor:
        """
        Applies soft thresholding to the input tensor.

        Args:
            x (Tensor): Input tensor.
            threshold (float): Threshold value for soft thresholding.

        Returns:
            Tensor: Tensor after applying soft thresholding.
        """
        return torch.sign(x) * torch.maximum(
            torch.abs(x) - threshold, torch.zeros_like(x)
        )


class SimbaBlock(nn.Module):
    """
    SimbaBlock is a module that represents a block in the Simba model.

    Args:
        dim (int): The input dimension.
        d_state (int, optional): The state dimension. Defaults to 64.
        d_conv (int, optional): The convolution dimension. Defaults to 64.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        num_classes (int, optional): The number of classes. Defaults to 10.

    Attributes:
        dim (int): The input dimension.
        d_state (int): The state dimension.
        d_conv (int): The convolution dimension.
        num_classes (int): The number of classes.
        dropout (nn.Dropout): The dropout layer.
        mamba (MambaBlock): The MambaBlock module.
        einfft (EinFFTText): The EinFFTText module.
        norm1 (nn.LayerNorm): The first layer normalization layer.
        norm2 (nn.LayerNorm): The second layer normalization layer.
    """

    def __init__(
            self,
            dim: int,
            d_state: int = 64,
            d_conv: int = 64,
            dropout: float = 0.1,
            num_classes: int = 10,
    ):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)

        self.mamba = MambaBlock(
            dim=self.dim,
            depth=1,
            d_state=self.d_state,
            d_conv=self.d_conv,
        )

        self.einfft = EinFFTText(
            sequence_length=num_classes,
            dim=self.dim,
        )

        self.norm1 = nn.LayerNorm(self.dim)
        self.norm2 = nn.LayerNorm(self.dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the SimbaBlock module.

        Args:
            x (Tensor): The input tensor of shape (batch_size, sequence_length, dim).

        Returns:
            Tensor: The output tensor of shape (batch_size, sequence_length, dim).
        """
        residual = x

        # Apply layer normalization and Mamba block
        x = self.norm1(x)
        x = self.mamba(x)
        x = self.dropout(x)

        # Add residual connection
        x = x + residual

        residual = x

        # Apply layer normalization and EinFFT
        x = self.norm2(x)
        x = self.einfft(x)
        x = self.dropout(x)

        # Add residual connection
        x = x + residual

        return x.real


class Simba(nn.Module):
    """
    Simba model implementation.

    Args:
        dim (int): Dimension of the model.
        num_classes (int): Number of output classes.
        depth (int, optional): Number of Simba blocks. Defaults to 8.
        d_state (int, optional): Dimension of the state. Defaults to 64.
        d_conv (int, optional): Dimension of the convolutional layer. Defaults to 64.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
        patch_size (int, optional): Size of the patches. Defaults to 16.
        image_size (int, optional): Size of the input image. Defaults to 224.
        channels (int, optional): Number of input channels. Defaults to 3.
        use_pos_emb (bool, optional): Whether to use positional embeddings. Defaults to True.

    Attributes:
        dim (int): Dimension of the model.
        num_classes (int): Number of output classes.
        depth (int): Number of Simba blocks.
        patch_size (int): Size of the patches.
        image_size (int): Size of the input image.
        use_pos_emb (bool): Whether to use positional embeddings.
        pos_emb (Tensor): Positional embeddings.
        to_patch (nn.Sequential): Patch embedding layer.
        to_latent (nn.Identity): Identity layer to convert to latent space.
        output_head (nn.Sequential): Output head layer.
        simba_blocks (nn.ModuleList): List of Simba blocks.
    """

    def __init__(
            self,
            dim: int,
            num_classes: int,
            depth: int = 8,
            d_state: int = 64,
            d_conv: int = 64,
            dropout: float = 0.1,
            patch_size: int = 16,
            image_size: int = 224,
            channels: int = 3,
            use_pos_emb: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.depth = depth
        self.patch_size = patch_size
        self.image_size = image_size
        self.use_pos_emb = use_pos_emb

        image_height, image_width = image_size, image_size
        patch_height, patch_width = patch_size, patch_size
        patch_dim = channels * patch_height * patch_width

        # Positional embeddings
        self.pos_emb = posemb_sincos_2d(
            h=image_height // patch_height,
            w=image_width // patch_width,
            dim=dim,
        )

        # Patch embedding layer
        self.to_patch = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, self.dim),
            nn.LayerNorm(dim),
        )

        # Identity layer to convert to latent space
        self.to_latent = nn.Identity()

        # Output head layer
        self.output_head = nn.Sequential(
            Reduce("b n d -> b d", "mean"),
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_classes),
        )

        # Simba blocks
        self.simba_blocks = nn.ModuleList(
            [
                SimbaBlock(
                    dim=self.dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    dropout=dropout,
                    num_classes=self.num_classes,
                )
                for _ in range(self.depth)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Simba model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Tensor: Output tensor of shape (batch_size, num_classes).
        """
        # Convert input to patches
        x = self.to_patch(x)

        # Add positional embeddings if enabled
        if self.use_pos_emb:
            x = x + self.pos_emb.to(x.device)

        # Pass through Simba blocks
        for block in self.simba_blocks:
            x = block(x)

        # Convert to latent space
        x = self.to_latent(x)

        # Pass through output head
        x = self.output_head(x)

        return x


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model parameters
    dim = 64
    num_classes = 10
    depth = 8
    d_state = 64
    d_conv = 64
    dropout = 0.1
    patch_size = 16
    image_size = 224
    channels = 3
    use_pos_emb = True

    # Create model
    model = Simba(
        dim=dim,
        num_classes=num_classes,
        depth=depth,
        d_state=d_state,
        d_conv=d_conv,
        dropout=dropout,
        patch_size=patch_size,
        image_size=image_size,
        channels=channels,
        use_pos_emb=use_pos_emb,
    ).to(device)

    # Print model summary
    print(model)

    # Generate random input
    batch_size = 1
    img = torch.randn(batch_size, channels, image_size, image_size).to(device)

    # Forward pass
    out = model(img)

    # Print output shape
    print(f"Output shape: {out.shape}")


if __name__ == "__main__":
    main()
