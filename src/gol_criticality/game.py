"""PyTorch-based Game of Life implementation."""
import torch
import torch.nn.functional as F


class GameOfLife:
    """GPU-accelerated Game of Life with configurable board size."""

    def __init__(
        self,
        board_size: tuple[int, int] = (64, 64),
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float16,
        init_density: float = 0.15,
    ):
        self.board_size = board_size
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.init_density = init_density

        self.kernel = torch.tensor(
            [[[[1, 1, 1],
               [1, 0, 1],
               [1, 1, 1]]]],
            dtype=self.dtype,
            device=self.device,
        )

    def init_state(self, batch_size: int = 1, init_density: float | None = None) -> torch.Tensor:
        """Initialize random board state."""
        if init_density is None:
            init_density = self.init_density
        shape = (batch_size, 1, self.board_size[0], self.board_size[1])
        return torch.bernoulli(
            torch.full(shape, init_density, dtype=self.dtype, device=self.device)
        )

    def step(self, state: torch.Tensor) -> torch.Tensor:
        """Advance simulation by one step."""
        with torch.no_grad():
            state_padded = F.pad(state, (1, 1, 1, 1), mode='constant', value=0)
            neighborhood_sum = F.conv2d(state_padded, self.kernel)
            new_state = torch.zeros_like(state, dtype=self.dtype, device=self.device)
            new_state = torch.where(neighborhood_sum == 3, 1, new_state)
            new_state = torch.where(neighborhood_sum == 2, state, new_state)
            return new_state

    @staticmethod
    def draw(state: torch.Tensor) -> "np.ndarray":
        """Convert state tensor to 2D numpy array for plotting."""
        import numpy as np
        return state[0, 0].cpu().numpy()
