import math
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
import math

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, TotalTokens, EmbeddingsDim):
        super().__init__()

        self.ImageSize = int(math.sqrt(TotalTokens-1))
        self.PositionalEncoding = nn.Parameter(
            torch.zeros(1, TotalTokens, EmbeddingsDim)
        )
        nn.init.trunc_normal_(self.PositionalEncoding, std=0.02)

    def forward(self, Image):
        TotalTokens = Image.size(1) - 1
        ImageSize = math.sqrt(TotalTokens)

        CLSPositionalEncoding = self.PositionalEncoding[:, 0:1, :]
        PositionalEncoding = self.PositionalEncoding[:, 1:, :].reshape(1, self.ImageSize, self.ImageSize, -1).permute(0, 3, 1, 2)
        PositionalEncoding = interpolate(PositionalEncoding, size = (int(ImageSize), int(ImageSize)), 
                                         mode ="bicubic", align_corners =False)
        PositionalEncoding = PositionalEncoding.flatten(2).permute(0, 2, 1)
        PositionalEncoding = torch.concat([CLSPositionalEncoding, PositionalEncoding], dim=1)

        return Image + PositionalEncoding
    
class AttentionPositionalEncoding(nn.Module):
    def __init__(self, TotalTokens, EmbeddingsDim):
        super().__init__()
        PositionalEncoding = torch.zeros(TotalTokens, EmbeddingsDim)
        Position = torch.arange(0, TotalTokens, dtype=torch.float32).unsqueeze(1)
        DivisionTerm = torch.exp(
            torch.arange(0, EmbeddingsDim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / EmbeddingsDim)
        )

        PositionalEncoding[:, 0::2] = torch.sin(Position * DivisionTerm)
        PositionalEncoding[:, 1::2] = torch.cos(Position * DivisionTerm)

        self.register_buffer("PositionalEncoding", PositionalEncoding.unsqueeze(0))

    def forward(self, Image):
        return Image + self.PositionalEncoding
    
def BuildPositionalEncoding(Type, TotalTokens, EmbeddingsDim):
    if Type == "Learnable":
        return LearnablePositionalEncoding(TotalTokens, EmbeddingsDim)
    elif Type == "Attention":
        return AttentionPositionalEncoding(TotalTokens, EmbeddingsDim)
    raise ValueError(f"Unknown Positional Encoding Type: {Type}") 