import torch.nn as nn

class DINOHead(nn.Module):
    def __init__(self, EmbeddingsDim= 256, DINOHiddenLayers = 2048, BottleNeckDim = 256):
        super().__init__()

        self.DINOHead = nn.Sequential(
            nn.Linear(EmbeddingsDim, DINOHiddenLayers),
            nn.GELU(),
            nn.Linear(DINOHiddenLayers, DINOHiddenLayers),
            nn.GELU(),
            nn.Linear(DINOHiddenLayers, BottleNeckDim),
            nn.LayerNorm(BottleNeckDim),
            nn.utils.weight_norm(nn.Linear(BottleNeckDim, BottleNeckDim**2))
        )

    def forward(self, Image):
        return self.DINOHead(Image)