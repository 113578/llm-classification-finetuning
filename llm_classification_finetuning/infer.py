"""Модуль для выполнения инференса на подготовленных данных.

Содержит точку входа infer, загружает модель и датамодуль, выполняет предсказания
и сохраняет CSV с предсказаниями.
"""

from pathlib import Path

import fire
import lightning as L
import pandas as pd
import torch
from hydra import compose, initialize

from llm_classification_finetuning.data import LCFDataModule
from llm_classification_finetuning.models import (
    DeepMLP,
    LCFModelModule,
    LogisticRegression,
)


def infer() -> None:
    """
    Выполняет инференс модели и сохраняет предсказания в submission.csv.

    Parameters
    ----------
    cfg : DictConfig
        Конфигурация Hydra.

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

        state_dict = torch.load(cfg.state_dict_file, map_location=cfg.device)

        model_module = LCFModelModule(model=model, lr=cfg.hyperparameters.lr)
        model_module.model.load_state_dict(state_dict=state_dict)
        model_module.model.to(device=cfg.device)
        model_module.model.eval()

        trainer = L.Trainer()
        predictions_batches = trainer.predict(model=model_module, datamodule=datamodule)
        predictions = torch.cat(tensors=[torch.tensor(batch) for batch in predictions_batches])

        df_pred = datamodule.df_pred.reset_index()

        one_hot_columns = list(datamodule.index2label.values())
        df_one_hot = pd.DataFrame(0, index=range(len(predictions)), columns=one_hot_columns)

        for i, index in enumerate(predictions):
            column_name = datamodule.index2label[index.item()]
            df_one_hot.loc[i, column_name] = 1

        outputs_path = Path(cfg.submission_file)
        outputs_path.parent.mkdir(parents=True, exist_ok=True)

        df_preds = pd.concat(objs=[df_pred['id'], df_one_hot], axis=1)
        df_preds.to_csv(path_or_buf=cfg.submission_file, index=False)


if __name__ == '__main__':
    fire.Fire(component=infer)
