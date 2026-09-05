import torch.nn as nn
from ViT.ViT import BuildModel
from DINOHead import DINOHead

import torch
import argparse
import yaml

class MultiCropWrapper(nn.Module):
    def __init__(self, ViTBackbone, DINOHead):
        super().__init__()

        self.ViTBackbone = ViTBackbone
        self.DINOHead = DINOHead

    def forward(self, image):
        Sizes = [i.shape[-2:] for i in image]
        UniqueSizes = sorted(set(Sizes), key= lambda s: Sizes.index(s))

        output = []
        for size in UniqueSizes:
            IndexOfSameSize = [i for i, s in enumerate(Sizes) if s == size]
            Batch = torch.concat([image[i] for i in IndexOfSameSize], dim=0)
            output.append(self.DINOHead(self.ViTBackbone(Batch)))

        return output

### Below Code was only for testing 

# def ParseArgs():
#     Parse = argparse.ArgumentParser()
#     Parse.add_argument("--Config", type= str, required= True)
#     return Parse.parse_args()
# def main():
#     Args = ParseArgs()
#     with open(Args.Config) as f:
#         Config = yaml.safe_load(f)

#     image = torch.rand(2, 3, 224, 224)
#     localimage = torch.rand(2, 3, 96, 96)

#     Image = []
#     for i in range(2):
#         Image.append(image)
#     for i in range(6):
#         Image.append(localimage)

#     print(len(Image))
#     print(Image[3].size(2))

#     backbone, head = BuildModel(Config, ForDINO = True), DINOHead(Config["EmbeddingsDim"], Config["DINOHiddenLayers"], Config["BottleNeckDim"])
#     student = MultiCropWrapper(ViTBackbone=backbone, DINOHead=head)
#     backbone2, head2 = BuildModel(Config, ForDINO = True), DINOHead(Config["EmbeddingsDim"], Config["DINOHiddenLayers"], Config["BottleNeckDim"])
#     teacher = MultiCropWrapper(ViTBackbone=backbone2, DINOHead= head2)
#     teacher(Image[:2])
#     student(Image)

# if __name__ == '__main__':
#     main()