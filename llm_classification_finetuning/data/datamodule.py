"""DataModule для подготовки данных и кодирования с помощью SentenceTransformers."""

import lightning as L
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from torch.utils.data import DataLoader

from llm_classification_finetuning.data.dataset import LCFDataset


class LCFDataModule(L.LightningDataModule):
    """
    Lightning DataModule для задачи классификации на embedding'ах.

    Parameters
    ----------
    train_file_path : str
        Путь к CSV с обучающими данными.
    pred_file_path : str | None
        Путь к CSV с данными для предсказания, может быть None.
    encoder_name : str
        Имя модели SentenceTransformer.
    batch_size : int
        Размер батча для DataLoader.
    train_size : float
        Доля обучающей выборки.
    val_size : float
        Доля валидационной выборки.
    device : str
        Устройство ('cpu', 'cuda', 'mps' и т.п.).
    random_state : int
        Random_state для split'а.
    num_workers : int
        Число воркеров для DataLoader.
    encoder_batch_size : int
        Размер батча для кодирования текстов.
    """

    def __init__(
        self,
        train_file_path: str,
        pred_file_path: str | None,
        encoder_name: str,
        batch_size: int,
        train_size: float,
        val_size: float,
        device: str,
        random_state: int,
        num_workers: int,
        encoder_batch_size: int,
    ):
        super().__init__()

        self.train_file_path = train_file_path
        self.pred_file_path = pred_file_path
        self.encoder_name = encoder_name
        self.batch_size = batch_size
        self.train_size = train_size
        self.val_size = val_size
        self.device = device
        self.random_state = random_state
        self.num_workers = num_workers
        self.encoder_batch_size = encoder_batch_size

        self.train_dataset = None
        self.val_dataset = None
        self.pred_dataset = None

        self.label_cols = ['winner_model_a', 'winner_model_b', 'winner_tie']
        self.label2index = {label: index for index, label in enumerate(self.label_cols)}
        self.index2label = {index: label for label, index in self.label2index.items()}

    def prepare_data(self) -> None:
        """
        Читает CSV'ы (train, pred) и добавляет вспомогательный столбец для стратификации.

        Returns
        -------
        None
        """
        self.df_full = pd.read_csv(
            filepath_or_buffer=self.train_file_path, usecols=[3, 4, 5, 6, 7, 8]
        )
        self.df_full['stratify_col'] = (
            self.df_full[['winner_model_a', 'winner_model_b', 'winner_tie']]
            .astype(str)
            .agg('_'.join, axis=1)
        )

        if self.pred_file_path is not None:
            self.df_pred = pd.read_csv(self.pred_file_path)
        else:
            self.df_pred = None

    def setup(self, stage: str | None = None) -> None:
        """
        Подготавливает данные для этапов 'fit' и 'predict', кодирует тексты и
        формирует LCFDataset для train, val и predict.

        Parameters
        ----------
        stage : str | None
            Этап ('fit', 'predict') или None.

        Returns
        -------
        None
        """
        encoder = SentenceTransformer(self.encoder_name, device=self.device)

        if stage == 'fit' or stage is None:
            df_train, df_val = train_test_split(
                self.df_full,
                train_size=self.train_size,
                test_size=self.val_size,
                random_state=self.random_state,
                stratify=self.df_full['stratify_col'],
            )

            train_embeddings = self._encode_df(df_train, encoder)
            val_embeddings = self._encode_df(df_val, encoder)

            train_labels = (
                torch.from_numpy(df_train[self.label_cols].to_numpy()).argmax(dim=1).long()
            )
            val_labels = torch.from_numpy(df_val[self.label_cols].to_numpy()).argmax(dim=1).long()

            self.train_dataset = LCFDataset(embeddings=train_embeddings, labels=train_labels)
            self.val_dataset = LCFDataset(embeddings=val_embeddings, labels=val_labels)

        if stage == 'predict':
            pred_embeddings = self._encode_df(self.df_pred, encoder)
            self.pred_dataset = LCFDataset(pred_embeddings, labels=None)

    def train_dataloader(self) -> DataLoader:
        """
        Возвращает DataLoader для train набора.

        Returns
        -------
        DataLoader
            Загрузчик обучающего набора.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Возвращает DataLoader для валидационного набора.

        Returns
        -------
        DataLoader
            Загрузчик валидационного набора.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        """
        Возвращает DataLoader для набора, на котором нужно сделать предсказания.

        Returns
        -------
        DataLoader
            Загрузчик набора предсказаний.
        """
        return DataLoader(
            self.pred_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def _encode_df(self, df: pd.DataFrame, encoder: SentenceTransformer) -> torch.Tensor:
        """
        Кодирует колонки prompt/response_a/response_b в embedding'и и формирует признаки.

        Parameters
        ----------
        df : pd.DataFrame
            Входной DataFrame с колонками 'prompt', 'response_a', 'response_b'.
        encoder : SentenceTransformer
            Модель для кодирования предложений.

        Returns
        -------
        torch.Tensor
            CPU-тензор с признаками для каждой пары (prompt, response_a, response_b).
        """
        prompt_embeddings = encoder.encode(
            sentences=df['prompt'].to_list(),
            batch_size=self.encoder_batch_size,
            convert_to_tensor=True,
            show_progress_bar=True,
        )

        response_a_embeddings = encoder.encode(
            sentences=df['response_a'].to_list(),
            batch_size=self.encoder_batch_size,
            convert_to_tensor=True,
            show_progress_bar=True,
        )

        response_b_embeddings = encoder.encode(
            sentences=df['response_b'].to_list(),
            batch_size=self.encoder_batch_size,
            convert_to_tensor=True,
            show_progress_bar=True,
        )

        prompt_embeddings = F.normalize(input=prompt_embeddings, dim=1)
        response_a_embeddings = F.normalize(input=response_a_embeddings, dim=1)
        response_a_embeddings = F.normalize(input=response_b_embeddings, dim=1)

        response_a_projection = (response_a_embeddings * prompt_embeddings).sum(
            dim=1, keepdim=True
        ) * prompt_embeddings
        response_b_projection = (response_b_embeddings * prompt_embeddings).sum(
            dim=1, keepdim=True
        ) * prompt_embeddings

        features = torch.cat(
            tensors=[
                response_a_embeddings - response_b_embeddings,
                response_a_projection - response_b_projection,
                (response_b_embeddings * prompt_embeddings).sum(dim=1, keepdim=True),
                (response_b_embeddings * prompt_embeddings).sum(dim=1, keepdim=True),
                (response_a_embeddings * response_b_embeddings).sum(dim=1, keepdim=True),
            ],
            dim=1,
        )

        return features.cpu()
