from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(eq=False, repr=False)
class MLP(nn.Module):
    dim_hidden: int = 64
    num_hidden_layers: int = 2

    def __post_init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, self.dim_hidden),
            nn.GELU(),
            *(
                [
                    nn.Linear(self.dim_hidden, self.dim_hidden),
                    nn.GELU(),
                ]
                * self.num_hidden_layers
            ),
            nn.Linear(self.dim_hidden, 2),
        )

    def forward(self, x: torch.Tensor, t: int):
        return self.net(torch.cat((x, torch.full((x.shape[0], 1), t)), dim=-1))
