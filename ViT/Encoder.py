import torch
import torch.nn as nn
from .Attention import MultiHeadAttention

class MLPBlock(nn.Module):
    def __init__(self, EmbeddingsDim, MLPRatio= 4, Dropout= 0.0):
        super().__init__()
        HiddenLayers = int(EmbeddingsDim * MLPRatio)
        self.MLP = nn.Sequential(
            nn.Linear(EmbeddingsDim, HiddenLayers),
            nn.GELU(),
            nn.Dropout(Dropout),
            nn.Linear(HiddenLayers, EmbeddingsDim),
            nn.Dropout(Dropout)

        )

    def forward(self, Image):
        return self.MLP(Image)
    

class Encoder(nn.Module):
    def __init__(self, EmbeddingsDim, NumHeads, Dropout = 0.0, MLPRatio = 4):
        super().__init__()
        self.LayerNorm1 = nn.LayerNorm(EmbeddingsDim)
        self.MultiHeadAttention = MultiHeadAttention(EmbeddingsDim, NumHeads, Dropout, Dropout)
        self.LayerNorm2 = nn.LayerNorm(EmbeddingsDim)
        self.MLP = MLPBlock(EmbeddingsDim, MLPRatio, Dropout)

    def forward(self, Image, ReturnAttention= False):
        if ReturnAttention:
            AttentionMap, AttentionProbs = self.MultiHeadAttention(self.LayerNorm1(Image), ReturnAttention)
            Image = Image + AttentionMap
            Image = Image + self.MLP(self.LayerNorm2(Image))
            return Image, AttentionProbs
        Image = Image + self.MultiHeadAttention(self.LayerNorm1(Image))
        Image = Image + self.MLP(self.LayerNorm2(Image))

        return Image
