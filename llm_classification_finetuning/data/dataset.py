import pandas as pd
import torch
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer


class LCFDataset(Dataset):
    def __init__(self, df: pd.DataFrame, encoder_name: str, device: str):
        self.df = df
        self.encoder = SentenceTransformer(
            model_name_or_path=encoder_name, device=device
        )

        prompts = self.encoder.encode(
            sentences=self.df["prompt"].to_list(),
            prompt="prompt",
            convert_to_tensor=True,
            device=device,
        )
        responses_a = self.encoder.encode(
            sentences=self.df["response_a"].to_list(),
            prompt="response",
            convert_to_tensor=True,
            device=device,
        )
        responses_b = self.encoder.encode(
            sentences=self.df["response_b"].to_list(),
            prompt="prompt",
            convert_to_tensor=True,
            device=device,
        )

        self.embeddings = torch.concatenate(
            tensors=[prompts, responses_a, responses_b], axis=1
        )
        self.labels = torch.tensor(
            data=self.df[["winner_model_a", "winner_model_b", "winner_tie"]].values,
            dtype=torch.long,
        )
        self.labels = torch.argmax(input=self.labels, dim=1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        return self.embeddings[index], self.labels[index]
