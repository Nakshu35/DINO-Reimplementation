import yaml
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from DINOHead import DINOHead
from MultiCropWrapper import MultiCropWrapper
from ViT.ViT import BuildModel
from DINOLoss import DINOLoss
from DATA.Dataset import Imagenette

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def ParseArgs():
    Parse = argparse.ArgumentParser()
    Parse.add_argument("--Config", type= str, required= True)
    return Parse.parse_args()

def main():
    Args = ParseArgs()
    with open(Args.Config) as f:
        Config = yaml.safe_load(f)

    Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = DINOAugementation(GlobalScale=(0.4, 1.0), LocalScale=(0.05, 0.4), NoLocalCrops=6)
    Data = Imagenette(RootDir="D:\\DLCV\\Representation Learning\\PROJECT\\DINO", transform=transform)
    Train = DataLoader(dataset=Data, batch_size=2, shuffle=True)

    Loss = DINOLoss(NCrops = 8, OutputDim = Config["OutputDim"])
    Loss.to(Device)

    StudentBackbone, StudentHead = BuildModel(Config=Config, ForDINO= True), DINOHead(EmbeddingsDim = Config['EmbeddingsDim'],
                                                                       DINOHiddenLayers = Config["DINOHiddenLayers"], 
                                                                       BottleNeckDim = Config["BottleNeckDim"],
                                                                        OutputDim = Config["OutputDim"])
    TeacherBackbone, TeacherHead = BuildModel(Config=Config, ForDINO= True), DINOHead(EmbeddingsDim = Config['EmbeddingsDim'],
                                                                           DINOHiddenLayers = Config["DINOHiddenLayers"], 
                                                                           BottleNeckDim = Config["BottleNeckDim"], 
                                                                            OutputDim = Config["OutputDim"])

    Student = MultiCropWrapper(ViTBackbone =StudentBackbone, DINOHead= StudentHead)
    Teacher = MultiCropWrapper(ViTBackbone= TeacherBackbone, DINOHead= TeacherHead)
    Student.to(Device)
    Teacher.to(Device)
    Teacher.load_state_dict(Student.state_dict())
    for Param in Teacher.parameters():
        Param.requires_grad = False
    Student.train()
    Teacher.eval()

    Optimizer = torch.optim.AdamW(Student.parameters(), lr=3e-4)

    for epoch in range(Config["Epochs"]):
        if epoch == 1:
            Student.DINOHead.LastLayer.weight_g.requires_grad = True
        for images, _ in Train:

            images = [image.to(Device) for image in images]

            Studentfeatures = Student(images)
            Teacherfeatures = Teacher(images[:2])

            Studentfeatures = torch.concat(Studentfeatures, dim=0)
            Teacherfeatures = torch.concat(Teacherfeatures, dim=0)

            TotalLoss = Loss(Studentfeatures, Teacherfeatures)

            Optimizer.zero_grad()
            TotalLoss.backward()
            Optimizer.step()

            with torch.no_grad():
                for StudentParam, TeacherParam in zip(Student.parameters(), Teacher.parameters()):
                    TeacherParam.data.mul_(Config["Momentum"])
                    TeacherParam.data.add_((1 - Config["Momentum"]) * StudentParam.data)




class DINOAugementation(object):
    def __init__(self, GlobalScale, LocalScale, NoLocalCrops):
          Normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

          self.GlobalTransform1 = transforms.Compose([
              transforms.RandomResizedCrop(224, scale=GlobalScale), 
              transforms.ToTensor(),
              Normalize
          ])
          self.GlobalTransform2 = transforms.Compose([
              transforms.RandomResizedCrop(224, scale=GlobalScale), 
              transforms.ToTensor(),
              Normalize
          ])

          self.NoLocalCrops = NoLocalCrops
          self.LocalTransform = transforms.Compose([
              transforms.RandomResizedCrop(96, LocalScale), 
              transforms.ToTensor(),
              Normalize
          ])

    def __call__(self, image):
        Crops = []
        Crops.append(self.GlobalTransform1(image))
        Crops.append(self.GlobalTransform2(image))
        for _ in range(self.NoLocalCrops):
            Crops.append(self.LocalTransform(image))

        return Crops


if __name__ == '__main__':
    main()