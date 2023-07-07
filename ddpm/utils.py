from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib.widgets import Slider
from torch.utils.data import TensorDataset

from .mlp import MLP
from .noise_scheduler import NoiseScheduler


def get_dataset(
    data_size: int = 1000,
    csv_file: Path = Path(__file__).parent.parent / "assets/shape/cat.csv",
    noise: float = 0.01,
) -> TensorDataset:
    df = pd.read_csv(csv_file)
    x = df["x"]
    y = df["y"]
    # normalize, adjusted, etc
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    # randomize order
    idx = np.random.randint(0, len(df), data_size)
    x = x.iloc[idx]
    y = y.iloc[idx]
    # add noise
    x += np.random.normal(size=len(x)) * noise
    y += np.random.normal(size=len(y)) * noise
    # create dataset
    X = np.stack((x, y), axis=1)
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


# assuming hydar
def reconstruct(log_dir_str: str) -> tuple[NoiseScheduler, MLP]:
    log_dir = Path(log_dir_str)
    cfg = yaml.safe_load(open(log_dir / ".hydra" / "config.yaml", "r"))
    ns = NoiseScheduler(
        **dict(filter(lambda x: x[0] != "_target_", cfg["noise_scheduler"].items()))
    )
    model = MLP(**dict(filter(lambda x: x[0] != "_target_", cfg["model"].items())))
    model.load_state_dict(torch.load(log_dir / "params.pt"))
    return ns, model


@torch.no_grad()
def denoise(log_dir_str: str, eval_data_size: int = 1000) -> list[torch.Tensor]:
    ns, model = reconstruct(log_dir_str)
    model.eval()
    x_last = torch.randn(eval_data_size, 2)
    samples = [x_last]
    for t in reversed(range(ns.num_timesteps)):
        residual = model(samples[-1], t)
        samples.append(ns.remove_noise(samples[-1], residual, t))
    return samples


def viz_sample(data, alpha: float = 0.3, figsize: int = 4, l: float = 2.5) -> None:
    x = data[:, 0]
    y = data[:, 1]
    plt.figure(figsize=(figsize, figsize))
    plt.scatter(x, y, s=3, alpha=alpha)
    plt.xlim(-l, l)
    plt.ylim(-l, l)
    plt.xticks([-l, 0, l])
    plt.yticks([-l, 0, l])
    plt.show()


# interactive plot
def viz_samples(samples, alpha: float = 0.3, figsize: int = 4, l: float = 2.5) -> None:
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    ax.set_xlim(-l, l)
    ax.set_ylim(-l, l)
    ax.set_xticks([-l, 0, l])
    ax.set_yticks([-l, 0, l])
    fig.subplots_adjust(bottom=0.22)
    scatter = ax.scatter(samples[0][:, 0], samples[0][:, 1], alpha=alpha, s=3)
    slider = Slider(
        fig.add_axes([0.2, 0.1, 0.65, 0.03]),
        label="",
        valmin=0,
        valmax=len(samples),
        valstep=range(0, len(samples)),
    )
    slider.on_changed(lambda _: scatter.set_offsets(samples[slider.val]))
    plt.show()
