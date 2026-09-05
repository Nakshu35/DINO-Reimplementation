import torch
import torch.nn as nn

class PatchEmbeddings(nn.Module):
    def __init__(self, ImageSize = 224, PatchSize = 16, InChannels = 3, EmbeddingsDim = 256):
        super().__init__()
        assert ImageSize % PatchSize == 0, "img_size must be divisible by patch_size"

        self.PatchSize = PatchSize
        self.NumPatches = (ImageSize // PatchSize) ** 2
        self.LinearProjection = nn.Conv2d(InChannels, EmbeddingsDim, kernel_size= PatchSize, stride= PatchSize)

    def forward(self, Image):
        ProjectedImage = self.LinearProjection(Image)
        FlattenImage = ProjectedImage.flatten(2).transpose(1, 2)

        return FlattenImage
