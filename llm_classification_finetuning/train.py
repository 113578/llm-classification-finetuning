"""Модуль для запуска процесса обучения модели.

Содержит точку входа train, использующую Hydra для конфигурации. Загружает
датамодуль, модель и запускает обучение с логированием в TensorBoard и MLflow.
"""

from pathlib import Path

import fire
import lightning as L
import torch
from hydra import compose, initialize
from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger

from llm_classification_finetuning.data import LCFDataModule
from llm_classification_finetuning.models import (
    DeepMLP,
    LCFModelModule,
    LogisticRegression,
)


def train() -> None:
    """
    Запускает процесс обучения модели.

    Parameters
    ----------
    cfg : DictConfig
        Конфигурация Hydra с разделами для data, model, training и т.д.

    Returns
    -------
    None
    """
    with initialize(config_path='../configs', version_base=None):
        cfg = compose(config_name='config')

        datamodule = LCFDataModule(
            train_file_path=cfg.data.train_file_path,
            pred_file_path=cfg.data.pred_file_path,
            encoder_name=cfg.data.encoder_name,
            batch_size=cfg.data.batch_size,
            train_size=cfg.data.train_size,
            val_size=cfg.data.val_size,
            device=cfg.device,
            random_state=cfg.random_state,
            num_workers=cfg.data.num_workers,
            encoder_batch_size=cfg.data.encoder_batch_size,
        )

        if cfg.model.type == 'baseline':
            model = LogisticRegression(in_features=cfg.in_features, num_classes=cfg.num_classes)
        elif cfg.model.type == 'deep_mlp':
            model = DeepMLP(
                in_features=cfg.in_features,
                num_classes=cfg.num_classes,
                hidden_dim=cfg.model.hidden_dim,
                depth=cfg.model.depth,
            )

        model_module = LCFModelModule(model=model, lr=cfg.hyperparameters.lr)

        tb_logger = TensorBoardLogger(
            save_dir='tensorboard_logs', name=cfg.model.type, log_graph=True
        )
        mlflow_logger = MLFlowLogger(
            tracking_uri=cfg.logging.tracking_uri, experiment_name=cfg.logging.experiment_name
        )

        trainer = L.Trainer(
            max_epochs=cfg.hyperparameters.num_epochs, logger=[tb_logger, mlflow_logger]
        )
        trainer.fit(model=model_module, datamodule=datamodule)

        mlflow_logger.log_hyperparams(
            params={'num_epochs': cfg.hyperparameters.num_epochs, 'lr': cfg.hyperparameters.lr}
        )
        mlflow_logger.log_metrics(metrics=trainer.logged_metrics)

        outputs_path = Path(cfg.state_dict_file)
        outputs_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(obj=model_module.model.state_dict(), f=cfg.state_dict_file)


if __name__ == '__main__':
    fire.Fire(component=train)
