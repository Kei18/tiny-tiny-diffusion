from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import call
from tqdm import tqdm


@hydra.main(
    version_base=None,
    config_path="train_conf",
    config_name="example",
)
def main(cfg) -> None:
    log_dir = Path(hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"])

    # setup
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    dataset = call(cfg.dataset)
    dataloader = call(cfg.dataloader, dataset)
    ns = call(cfg.noise_scheduler)
    model = call(cfg.model)
    optimizer = call(cfg.optimizer, model.parameters())
    criterion = torch.nn.MSELoss(reduction="sum")

    # training
    with tqdm(total=cfg.num_epochs) as pbar:
        for _ in range(cfg.num_epochs):
            model.train()
            train_loss = 0
            for batch in dataloader:
                batch = batch[0]
                t = np.random.randint(0, ns.num_timesteps)
                epsilon_target = torch.randn(batch.shape)  # noise
                x_t_plus_1 = ns.add_noise(batch, epsilon_target, t)  # x_{t+1}
                epsilon_pred = model(x_t_plus_1, t)  # predicted noise
                loss = criterion(epsilon_pred, epsilon_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(dataset)
            pbar.set_postfix({"loss": f"{train_loss:.4f}"})
            pbar.update(1)

    torch.save(model.state_dict(), log_dir / "params.pt")
    print(f"saving model to {log_dir}")


if __name__ == "__main__":
    main()
