from __future__ import annotations
import copy
import random
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SharedEncoder(nn.Module):
    """The same two-layer encoder is used by every neural model family."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QuantileNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        taus: np.ndarray,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = SharedEncoder(input_dim, hidden_dim, dropout)
        self.head = nn.Linear(hidden_dim, len(taus))
        self.register_buffer("taus", torch.as_tensor(taus, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sort(self.head(self.encoder(x)), dim=1).values

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        errors = y[:, None] - self(x)
        return torch.maximum(self.taus * errors, (self.taus - 1) * errors).mean()


class GMMNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        components: int = 10,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.components = components
        self.encoder = SharedEncoder(input_dim, hidden_dim, dropout)
        self.mean = nn.Linear(hidden_dim, components)
        self.log_scale = nn.Linear(hidden_dim, components)
        self.logits = nn.Linear(hidden_dim, components)

    def parameters_for(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return (
            self.mean(z),
            nn.functional.softplus(self.log_scale(z)) + 0.0001,
            self.logits(z),
        )

    def distribution(self, x: torch.Tensor) -> torch.distributions.MixtureSameFamily:
        (mean, scale, logits) = self.parameters_for(x)
        return torch.distributions.MixtureSameFamily(
            torch.distributions.Categorical(logits=logits),
            torch.distributions.Normal(mean, scale),
        )

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return -self.distribution(x).log_prob(y).mean()


class LinearSplineNet(nn.Module):
    """SPICE n=1 density head with learned conditional knot positions."""

    def __init__(
        self,
        input_dim: int,
        knots: int = 21,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        if knots < 3:
            raise ValueError("knots must be at least 3")
        self.knots = knots
        self.encoder = SharedEncoder(input_dim, hidden_dim, dropout)
        self.head = nn.Sequential(nn.GELU(), nn.Linear(hidden_dim, knots))
        self.width_head = nn.Sequential(nn.GELU(), nn.Linear(hidden_dim, knots - 1))
        inverse_one = float(1.0 + np.log(-np.expm1(-1.0)))
        nn.init.constant_(self.head[-1].bias, inverse_one)
        nn.init.normal_(self.head[-1].weight, std=0.1)
        nn.init.constant_(self.width_head[-1].bias, -np.log(knots - 1))
        nn.init.normal_(self.width_head[-1].weight, std=0.1)

    def positions_heights(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        h = nn.functional.softplus(self.head(z)).clamp_min(0.01)
        minimum_width = 1.0 / (self.knots * 10.0)
        widths = nn.functional.softmax(self.width_head(z), dim=1)
        widths = minimum_width + (1.0 - minimum_width * self.knots) * widths
        positions = torch.cat(
            (torch.zeros_like(widths[:, :1]), torch.cumsum(widths, dim=1)), dim=1
        )
        positions[:, -1] += minimum_width
        dx = positions[:, 1:] - positions[:, :-1]
        integral = (0.5 * (h[:, :-1] + h[:, 1:]) * dx).sum(1, keepdim=True)
        return (positions, h / integral)

    def heights(self, x: torch.Tensor) -> torch.Tensor:
        return self.positions_heights(x)[1]

    def density_at(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        (positions, h) = self.positions_heights(x)
        y = y.clamp(0, 1 - 1e-07)
        idx = torch.clamp(
            (y[:, None] >= positions[:, 1:-1]).sum(dim=1), 0, self.knots - 2
        )
        left = positions.gather(1, idx[:, None]).squeeze(1)
        right = positions.gather(1, (idx + 1)[:, None]).squeeze(1)
        (hl, hr) = (
            h.gather(1, idx[:, None]).squeeze(1),
            h.gather(1, (idx + 1)[:, None]).squeeze(1),
        )
        return hl + (hr - hl) * (y - left) / (right - left)

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return -torch.log(self.density_at(x, y).clamp_min(1e-12)).mean()


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 1e-05
    patience: int = 15
    device: str = "auto"


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainConfig,
    seed: int,
) -> dict[str, float | int]:
    set_seed(seed)
    device = resolve_device(config.device)
    model.to(device)
    generator = torch.Generator().manual_seed(seed)
    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader = DataLoader(
        train_data, batch_size=config.batch_size, shuffle=True, generator=generator
    )
    (xv, yv) = (torch.from_numpy(X_val).to(device), torch.from_numpy(y_val).to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_state = copy.deepcopy(model.state_dict())
    (best_loss, best_epoch, stale) = (float("inf"), 0, 0)
    train_history: list[float] = []
    val_history: list[float] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_total = 0.0
        epoch_count = 0
        for (xb, yb) in loader:
            (xb, yb) = (xb.to(device), yb.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(xb, yb)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            epoch_total += float(loss.detach().cpu()) * len(xb)
            epoch_count += len(xb)
        model.eval()
        with torch.no_grad():
            val_loss = float(model.loss(xv, yv).cpu())
        train_history.append(epoch_total / epoch_count)
        val_history.append(val_loss)
        if val_loss < best_loss - 1e-07:
            (best_loss, best_epoch, stale) = (val_loss, epoch, 0)
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= config.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return {
        "best_epoch": best_epoch,
        "validation_loss": best_loss,
        "parameters": sum((p.numel() for p in model.parameters())),
        "train_history": train_history,
        "validation_history": val_history,
        "device": str(device),
    }


def predict_quantiles(
    model: QuantileNet, X: np.ndarray, batch_size: int = 2048
) -> np.ndarray:
    device = next(model.parameters()).device
    output = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            output.append(
                model(torch.from_numpy(X[start : start + batch_size]).to(device))
                .cpu()
                .numpy()
            )
    return np.concatenate(output)


def density_grid(
    model: nn.Module, X: np.ndarray, grid: np.ndarray, batch_size: int = 512
) -> np.ndarray:
    device = next(model.parameters()).device
    grid_t = torch.as_tensor(grid, dtype=torch.float32, device=device)
    output = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[start : start + batch_size]).to(device)
            if isinstance(model, GMMNet):
                dist = model.distribution(xb)
                values = dist.log_prob(grid_t[:, None]).exp().T
            elif isinstance(model, LinearSplineNet):
                (positions, h) = model.positions_heights(xb)
                idx = torch.clamp(
                    (grid_t[None, :, None] >= positions[:, None, 1:-1]).sum(dim=2),
                    0,
                    model.knots - 2,
                )
                (left, right) = (positions.gather(1, idx), positions.gather(1, idx + 1))
                (hl, hr) = (h.gather(1, idx), h.gather(1, idx + 1))
                values = hl + (hr - hl) * (grid_t[None, :] - left) / (right - left)
            else:
                raise TypeError(type(model))
            output.append(values.cpu().numpy())
    return np.concatenate(output)


def density_at(
    model: nn.Module, X: np.ndarray, y: np.ndarray, batch_size: int = 2048
) -> np.ndarray:
    device = next(model.parameters()).device
    output = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[start : start + batch_size]).to(device)
            yb = torch.from_numpy(y[start : start + batch_size]).to(device)
            if isinstance(model, GMMNet):
                values = model.distribution(xb).log_prob(yb).exp()
            elif isinstance(model, LinearSplineNet):
                values = model.density_at(xb, yb)
            else:
                raise TypeError(type(model))
            output.append(values.cpu().numpy())
    return np.concatenate(output)


def predict_spline_knots(
    model: LinearSplineNet, X: np.ndarray, batch_size: int = 2048
) -> tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    (positions, heights) = ([], [])
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            (p, h) = model.positions_heights(
                torch.from_numpy(X[start : start + batch_size]).to(device)
            )
            positions.append(p.cpu().numpy())
            heights.append(h.cpu().numpy())
    return (np.concatenate(positions), np.concatenate(heights))
