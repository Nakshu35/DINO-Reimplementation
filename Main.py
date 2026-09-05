import yaml
import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import csv

from DINOHead import DINOHead
from MultiCropWrapper import MultiCropWrapper
from ViT.ViT import BuildModel
from DINOLoss import DINOLoss
from DATA.Dataset import Imagenette
from Engine.Train import Train

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

    ExperimentName = Config["ExperimentName"]
    SaveDir = os.path.join(Config["RootDir"], "Saves", ExperimentName)
    os.makedirs(SaveDir, exist_ok=True)
    CsvPath = os.path.join(SaveDir, f"{ExperimentName}Train.csv")
    with open(CsvPath, 'w', newline="") as f:
        csv.writer(f).writerow(["Epoch", "Train Loss", "Lr"])

    TrainAugmentation = DINOAugementation(GlobalScale=(0.4, 1.0), LocalScale=(0.05, 0.4), NoLocalCrops=6)
    TrainData = Imagenette(RootDir=Config["RootDir"], transform=TrainAugmentation)
    TrainSet = DataLoader(dataset=TrainData, batch_size=Config["BatchSize"], shuffle=True)

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

    Optimizer = torch.optim.AdamW(Student.parameters(), lr=3e-4)
    
    BestLoss = -float('inf')
    with open(CsvPath, "a", newline="") as f:
        Writer = csv.writer(f)
        for epoch in range(Config["Epochs"]):
            if epoch == 1:
                Student.DINOHead.LastLayer.weight_g.requires_grad = True
            TrainLoss, CurrentLr = Train(Student, Teacher, Loss, Optimizer, Device, TrainSet, Config)
            print(f"Epoch {epoch+1} TrainLoss = {TrainLoss:.4f}")

            Writer.writerow([epoch+1, f"{TrainLoss:.4f}", CurrentLr])

            if TrainLoss > BestLoss:
                BestLoss = TrainLoss
                torch.save(Student.state_dict(), os.path.join(SaveDir, "Student.pt"))
                torch.save(Teacher.state_dict(), os.path.join(SaveDir, "Teacher.pt"))
                print("Best Model saved")


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
