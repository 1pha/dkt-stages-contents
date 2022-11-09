import os

import hydra
import omegaconf
import torch
import wandb

from code.dkt.src import trainer
from code.dkt.src.dataloader import Preprocess
from code.dkt.src.utils import setSeeds


@hydra.main(version_base="1.2", config_path="configs", config_name="default")
def main(config: omegaconf.DictConfig = None) -> None:

    omegaconf.OmegaConf.set_struct(config, False)

    wandb.login()
    setSeeds(config.seed)
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(config.model_dir, exist_ok=True)

    preprocess = Preprocess(config.data)
    preprocess.load_train_data(config.data.file_name)
    train_data = preprocess.get_train_data()

    train_data, valid_data = preprocess.split_data(train_data)

    wandb.init(project="dkt", config=vars(config))

    # Instantiate model with hydra
    model = hydra.utils.instantiate(config.model).to(config.device)
    trainer.run(config, train_data, valid_data, model)


if __name__ == "__main__":
    main()
