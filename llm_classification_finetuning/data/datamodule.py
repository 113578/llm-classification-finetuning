import lightning as L
import pandas as pd
from llm_classification_finetuning.data.dataset import LCFDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


class LCFDataModule(L.LightningDataModule):
    def __init__(
        self,
        file_path: str,
        encoder_name: str,
        batch_size: int,
        train_size: float,
        val_size: float,
        test_size: float,
        device: str,
        random_state: int,
    ):
        super().__init__()

        self.df = pd.read_csv(filepath_or_buffer=file_path, usecols=[3, 4, 5, 6, 7, 8])
        self.encoder_name = encoder_name
        self.batch_size = batch_size

        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

        self.device = device
        self.random_state = random_state

    def setup(self, stage: str = None):
        self.df["stratify_column"] = (
            self.df[["winner_model_a", "winner_model_b", "winner_tie"]]
            .astype(dtype=str)
            .agg(func="_".join, axis=1)
        )

        df_train_val, df_test = train_test_split(
            self.df,
            train_size=self.train_size,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.df["stratify_column"],
        )
        df_train, df_val = train_test_split(
            df_train_val,
            train_size=self.train_size,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=df_train_val["stratify_column"],
        )

        self.train_dataset = LCFDataset(
            df=df_train, encoder_name=self.encoder_name, device=self.device
        )
        self.val_dataset = LCFDataset(
            df=df_val, encoder_name=self.encoder_name, device=self.device
        )
        self.test_dataset = LCFDataset(
            df=df_test, encoder_name=self.encoder_name, device=self.device
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False
        )
