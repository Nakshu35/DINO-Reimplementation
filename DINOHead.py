import torch.nn as nn

class DINOHead(nn.Module):
    def __init__(self, EmbeddingsDim= 256, DINOHiddenLayers = 2048, BottleNeckDim = 256, OutputDim= 65536):
        super().__init__()

        self.MLP = nn.Sequential(
            nn.Linear(EmbeddingsDim, DINOHiddenLayers),
            nn.GELU(),
            nn.Linear(DINOHiddenLayers, DINOHiddenLayers),
            nn.GELU(),
            nn.Linear(DINOHiddenLayers, BottleNeckDim),
        )
        self.LastLayer = nn.utils.weight_norm(nn.Linear(BottleNeckDim, OutputDim, bias= False))
        self.LastLayer.weight_g.requires_grad = False

    def forward(self, Image):
        Image = self.MLP(Image)
        Image = nn.functional.normalize(Image, dim=1)
        Image = self.LastLayer(Image)
        return Image