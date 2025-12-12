import torch.nn as nn


class LogisticRegression(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()

        self.model = nn.Linear(in_features=in_features, out_features=num_classes)

    def forward(self, embeddings):
        return self.model(embeddings)
