"""
Equations are based on:
Luo, C. Understanding diffusion models: A unified perspective. arXiv preprint arXiv:2208.11970. 2022.
"""


from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(eq=False, repr=False)
class NoiseScheduler:
    beta_start: float = 1e-5
    beta_end: float = 1e-2
    num_timesteps: int = 50

    def __post_init__(self) -> None:
        super().__init__()
        # compute parameters
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = nn.functional.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.0
        )
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.sqrt_inv_alphas = torch.sqrt(1 / self.alphas)

        # parameters for forward process
        self.coef_noise_mu = self.sqrt_alphas_cumprod
        self.coef_noise_sigma = self.sqrt_one_minus_alphas_cumprod

        # parameters for backward process
        self.coef_denoise_mu_1 = self.sqrt_inv_alphas
        self.coef_denoise_mu_2 = (
            self.betas / torch.sqrt(1 - self.alphas_cumprod) * self.sqrt_inv_alphas
        )
        self.coef_denoise_sigma = torch.sqrt(
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def add_noise(
        self, x_0: torch.Tensor, epsilon: torch.Tensor, t: int
    ) -> torch.Tensor:
        """
        sampling x_t ~ N(mu, sigma^2) with reparameterization trick

        mu (Eq. 70): \sqrt{\bar{a}_t} * x_0

        sigma^2 (Eq. 70): (1 - \bar{a}_t)
        """
        mu = self.coef_noise_mu[t] * x_0
        sigma = self.coef_noise_sigma[t]
        return mu + epsilon * sigma

    def remove_noise(
        self, x_t: torch.Tensor, epsilon_pred: torch.Tensor, t: int
    ) -> torch.Tensor:
        """
        sampling x_{t-1} ~ N(mu, sigma^2) with reparameterization trick

        mu (Eq. 125): (1 / \sqrt{a_t}) * x_t - [(1 - a_t) / \sqrt{a_t} / \sqrt{1 - \bar{a}_t}] * \epsilon

        sigma^2 (Eq. 85): (1 - a_t) * (1 - \bar{a}_{t - 1}) / (1 - \bar{a}_t)
        """
        mu_x_t = (
            self.coef_denoise_mu_1[t] * x_t - self.coef_denoise_mu_2[t] * epsilon_pred
        )
        sigma_t = self.coef_denoise_sigma[t]
        z = torch.randn_like(epsilon_pred)
        return mu_x_t + sigma_t * z
