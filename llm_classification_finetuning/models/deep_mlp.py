import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dimension: int, dropout_probability: float):
        super().__init__()

        self.fc1 = nn.Linear(in_features=dimension, out_features=dimension)
        self.fc2 = nn.Linear(in_features=dimension, out_features=dimension)
        self.layer_norm = nn.LayerNorm(normalized_shape=dimension)
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, embeddings):
        residual = embeddings

        embeddings = self.fc1(embeddings)
        embeddings = F.gelu(embeddings)
        embeddings = self.dropout(embeddings)
        embeddings = self.fc2(embeddings)

        embeddings = self.norm(embeddings + residual)

        return embeddings


class DeepMLP(nn.Module):
    def __init__(self, in_features: int, num_classes: int, hidden_dim: int, depth: int):
        super().__init__()

        self.input_projection = nn.Linear(
            in_features=in_features, out_features=hidden_dim
        )
        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_dim)

        self.blocks = nn.Sequential(
            *[
                ResidualBlock(dimension=hidden_dim, dropout_probability=0.1)
                for _ in range(depth)
            ]
        )

        self.output_layer = nn.Linear(in_features=hidden_dim, out_features=num_classes)

    def forward(self, embeddings):
        embeddings = self.input_projection(embeddings)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.blocks(embeddings)
        embeddings = self.output_layer(embeddings)

        return embeddings
