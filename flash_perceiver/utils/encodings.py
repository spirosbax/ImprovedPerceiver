import torch
import torch.nn as nn

from abc import abstractmethod, ABC
from typing import Optional
from math import pi
from einops import rearrange


T = torch.Tensor


class BasePositionalEncoding(nn.Module, ABC):
    """Base class for positional encoders.

    An implementation set values for in_dim and out_dim.

    Attributes:
        in_dim: Expected input dimensionality of the encoder.
        out_dim: Output dimensionality of the encoder.
    """

    in_dim: int
    out_dim: int

    @abstractmethod
    def forward(self, x: T) -> T:
        """foo"""
        raise NotImplementedError


class FourierPositionalEncoding(BasePositionalEncoding):
    """Projects an input by the given projection matrix before applying a sinus function.
    The input will be concatenated along the last axis.

    Args:
        proj_matrix: Projection matrix of shape ``(m, n)``.
        is_trainable: Whether the projection should be stored as trainable parameter. Default: ``False``

    Raises:
        ValueError: Raised if the given projection matrix does not have two dimensions.
    """

    def __init__(self, proj_matrix: T, is_trainable: bool = False):
        super().__init__()

        if proj_matrix.ndim != 2:
            raise ValueError(
                f"Expected projection matrix to have two dimensions but found {proj_matrix.ndim}"
            )

        self.is_trainable = is_trainable

        if is_trainable:
            self.register_parameter("proj_matrix", nn.Parameter(proj_matrix))
        else:
            self.register_buffer("proj_matrix", proj_matrix)

        self.in_dim, self.out_dim = self.proj_matrix.shape

    def forward(self, x: T) -> T:
        channels = x.shape[-1]

        assert (
            channels == self.in_dim
        ), f"Expected input to have {self.in_dim} channels but found {channels} channels instead)"

        x = torch.einsum("... i, i j -> ... j", x, self.proj_matrix)
        x = 2 * pi * x

        return torch.sin(x)


class IdentityPositionalEncoding(BasePositionalEncoding):
    """Positional encoder that returns the identity of the input."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = in_dim

    def forward(self, x: T) -> T:
        return x


class GaussianFourierFeatureTransform(FourierPositionalEncoding):
    """Implements the positional encoder proposed in (Tancik et al., 2020).

    Args:
        in_dim: Dimensionality of inputs.
        mapping_size: Dimensionality to map inputs to. Default: ``32``
        sigma: SD of the gaussian projection matrix. Default: ``1.0``
        is_trainable: Whether the projection should be stored as trainable parameter. Default: ``False``
        seed: Optional seed for the random number generator.

    Attributes:
        in_dim: Expected input dimensionality.
        out_dim: Output dimensionality (mapping_size * 2).
    """

    def __init__(
        self,
        in_dim: int,
        mapping_size: int = 32,
        sigma: float = 1.0,
        is_trainable: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__(
            self.get_proj_matrix(in_dim, mapping_size, sigma, seed=seed),
            is_trainable=is_trainable,
        )
        self.mapping_size = mapping_size
        self.sigma = sigma
        self.seed = seed

    @classmethod
    def get_proj_matrix(cls, in_dim, mapping_size, sigma, seed=None):
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        return torch.normal(
            mean=0, std=sigma, size=(in_dim, mapping_size), generator=generator
        )

    @classmethod
    def from_proj_matrix(cls, projection_matrix):
        in_dim, mapping_size = projection_matrix.shape
        feature_transform = cls(in_dim, mapping_size)
        feature_transform.projection_matrix.data = projection_matrix
        return feature_transform


class NeRFPositionalEncoding(FourierPositionalEncoding):
    """Implements the NeRF positional encoding from (Mildenhall et al., 2020).

    Args:
        in_dim: Dimensionality of inputs.
        num_frequency_bands: Number of frequency bands where the i-th band has frequency :math:`f_{i} = 2^{i}`.
            Default: ``10``

    Attributes:
        in_dim: Expected input dimensionality.
        out_dim: Output dimensionality (in_dim * n * 2).
    """

    def __init__(self, in_dim: int, num_frequency_bands: int = 10):
        super().__init__((2.0 ** torch.arange(num_frequency_bands))[None, :])
        self.num_frequency_bands = num_frequency_bands
        self.out_dim = num_frequency_bands * 2 * in_dim

    def forward(self, x: T) -> T:
        x = rearrange(x, "... -> ... 1") * self.proj_matrix
        x = pi * x
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        x = rearrange(x, "... i j -> ... (i j)")
        return x


class PerceiverPositionalEncoding(BasePositionalEncoding):
    """Implements the Fourier-based positional encoding as described in the original Perceiver paper.

    This encoder maps input positions by:

    1. Linearly spacing frequency bands from 1.0 to (max_res / 2) (the Nyquist frequency).
    2. For each input dimension, computing sine and cosine features:
       $$\sin(\pi\, f\, x) \quad \text{and} \quad \cos(\pi\, f\, x),$$
       where \(f\) is a frequency band.
    3. Optionally concatenating the raw input positions.

    Args:
        in_dim: Dimensionality of input positions.
        n_bands: Number of frequency bands (K) to use.
        max_res: Maximum resolution (i.e. number of samples per dimension); the highest frequency becomes max_res/2.
        concat_pos: Whether to concatenate the raw input positions to the Fourier features. Default: True.

    Output:
        If `concat_pos` is True, the output will have dimension
        $$\text{in_dim} \times (1 + 2 \times \text{n_bands}),$$
        otherwise it will be $$2 \times \text{in_dim} \times \text{n_bands}.$$
    """

    def __init__(
        self, in_dim: int, n_bands: int, max_res: float, concat_pos: bool = True
    ):
        super().__init__()
        self.in_dim = in_dim
        self.n_bands = n_bands
        self.max_res = max_res
        self.concat_pos = concat_pos

        # Define frequency range from 1.0 to the Nyquist frequency, which is max_res / 2.
        min_freq = 1.0
        max_freq = max_res / 2
        freq_bands = torch.linspace(
            min_freq, max_freq, steps=n_bands, dtype=torch.float32
        )
        self.register_buffer("freq_bands", freq_bands)

        if concat_pos:
            self.out_dim = in_dim * (1 + 2 * n_bands)
        else:
            self.out_dim = 2 * in_dim * n_bands

    def forward(self, x: T) -> T:
        # x is assumed to have shape [..., in_dim]
        # Expand the last dimension to multiply with freq_bands; result becomes [..., in_dim, n_bands]
        pos_freq = x.unsqueeze(-1) * self.freq_bands  # broadcasting over last axis
        # Compute sine and cosine features; note the factor pi.
        sin_feats = torch.sin(pi * pos_freq)
        cos_feats = torch.cos(pi * pos_freq)
        # Flatten the last two dimensions: from [..., in_dim, n_bands] to [..., in_dim * n_bands]
        sin_feats_flat = rearrange(sin_feats, "... a b -> ... (a b)")
        cos_feats_flat = rearrange(cos_feats, "... a b -> ... (a b)")

        # Optionally concatenate the raw input positions.
        if self.concat_pos:
            encoding = torch.cat([x, sin_feats_flat, cos_feats_flat], dim=-1)
        else:
            encoding = torch.cat([sin_feats_flat, cos_feats_flat], dim=-1)

        return encoding


def get_encoder(name: str, in_dim: int, **kwargs):
    encoders = {
        "identity": IdentityPositionalEncoding,
        "gaussian_fourier_features": GaussianFourierFeatureTransform,
        "nerf": NeRFPositionalEncoding,
    }

    if name not in encoders:
        raise ValueError(f"Unknown encoder {name}. Must be one of {list(encoders)}.")

    return encoders[name](in_dim, **kwargs)
