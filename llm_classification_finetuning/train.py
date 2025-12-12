import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
import hydra
from omegaconf import DictConfig

from llm_classification_finetuning.data import LCFDataModule
from llm_classification_finetuning.models import LogisticRegression, DeepMLP
from llm_classification_finetuning.model_module import LCFModelModule


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train(cfg: DictConfig):
    datamodule = LCFDataModule(
        file_path=cfg.data.file_path,
        encoder_name=cfg.data.encoder_name,
        batch_size=cfg.data.batch_size,
        train_size=cfg.data.train_size,
        val_size=cfg.data.val_size,
        test_size=cfg.data.test_size,
        random_state=cfg.random_state,
        device=cfg.device,
    )

    if cfg.model.type == "baseline":
        model = LogisticRegression(
            in_features=cfg.in_features, num_classes=cfg.num_classes
        )
    elif cfg.model.type == "deep_mlp":
        model = DeepMLP(
            in_features=cfg.in_features,
            num_classes=cfg.num_classes,
            hidden_dim=cfg.model.hidden_dim,
            depth=cfg.model.depth,
        )

    model_module = LCFModelModule(model=model, lr=cfg.lr)

    logger = TensorBoardLogger(
        save_dir="tensor_board_logs", name=cfg.logging.model_name
    )

    trainer = L.Trainer(max_epochs=cfg.num_epochs, logger=logger)
    trainer.fit(model=model_module, datamodule=datamodule)

    torch.save(obj=model_module.model.state_dict(), f=cfg.output_file)


if __name__ == "__main__":
    train()
