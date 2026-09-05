import torch
import torch.nn as nn
from .Encoder import Encoder
from .PatchEmbeddings import PatchEmbeddings
from .PositionalEncoding import BuildPositionalEncoding

class ViT(nn.Module):
    def __init__(self, ImageSize= 224, PatchSize= 16, InChannels= 3, EmbeddingsDim= 256,
                 NumHeads= 8, MLPRatio= 4, Dropout= 0.0, NumClasses= 20, NumLayers= 6,
                 PositionalEncoding= "Learnable", Pooling= "CLS", ForDINO = False):
        super().__init__()

        assert Pooling in ("CLS", "Mean"), "Pooling must be 'CLS' or 'Mean'"
        self.Pooling = Pooling

        self.PatchEmbeddings = PatchEmbeddings(ImageSize, PatchSize, InChannels, EmbeddingsDim)
        self.NumPatches = self.PatchEmbeddings.NumPatches

        if Pooling == "CLS":
            self.CLSToken = nn.Parameter(torch.zeros(1, 1, EmbeddingsDim))
            nn.init.trunc_normal_(self.CLSToken, std= 0.02)
            self.NumPositions = self.NumPatches + 1
        else:
            self.CLSToken = None
            self.NumPositions = self.NumPatches

        self.NumPositions 
        self.PositionalEncoding = BuildPositionalEncoding(PositionalEncoding, self.NumPositions, EmbeddingsDim)
        self.PositionalDropout = nn.Dropout(Dropout)

        self.Blocks = nn.ModuleList([
            Encoder(EmbeddingsDim, NumHeads, Dropout, MLPRatio)
            for _ in range(NumLayers)
        ])
        self.LayerNorm = nn.LayerNorm(EmbeddingsDim)
        self.Head = nn.Linear(EmbeddingsDim, NumClasses)

        self.ForDINO = ForDINO

    def forward(self, Image, ReturnAttention= False):
        Image = self.PatchEmbeddings(Image)
        if self.CLSToken is not None:
            CLSToken = self.CLSToken.expand(Image.shape[0], -1, -1)
            Image = torch.concat([CLSToken, Image], dim= 1)
        Image = self.PositionalEncoding(Image)
        Image = self.PositionalDropout(Image)
        AttentionMaps = []

        for Block in self.Blocks:
            if ReturnAttention:
                Image, AttentionMap = Block(Image, ReturnAttention)
                AttentionMaps.append(AttentionMap)
            else:
                Image = Block(Image)

        Image = self.LayerNorm(Image)

        if self.Pooling == "CLS":
            Pooled = Image[:, 0]
        else:
            Pooled = Image.mean(dim=1)

        if self.ForDINO:
            return Pooled
        
        Output = self.Head(Pooled)

        if ReturnAttention:
            return Output, AttentionMaps
        else:return Output

def BuildModel(Config, ForDINO = False):
    return ViT(
        ImageSize= Config["ImageSize"],
        PatchSize= Config["PatchSize"],
        EmbeddingsDim= Config["EmbeddingsDim"],
        NumHeads= Config["NumHeads"],
        NumLayers= Config["NumLayers"],
        PositionalEncoding= Config["PositionalEncoding"],
        Pooling= Config["Pooling"], 
        ForDINO= ForDINO,
    )