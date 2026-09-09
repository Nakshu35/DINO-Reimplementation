import yaml
import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import csv
import torch.nn as nn
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
    TrainData = Imagenette(DatasetDir=Config["DatasetDir"], transform=TrainAugmentation)
    TrainSet = DataLoader(dataset=TrainData, batch_size=Config["BatchSize"], shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)

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
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        Student = nn.DataParallel(Student)
        Teacher = nn.DataParallel(Teacher)
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
                torch.save(Student, os.path.join(SaveDir, "Student.pt"))
                torch.save(Teacher, os.path.join(SaveDir, "Teacher.pt"))
                print("Best Model saved")


class DINOAugementation(object):
    def __init__(self, GlobalScale, LocalScale, NoLocalCrops):
        Normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        ColorJitter = transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
            p=0.8
        )
        GrayScale = transforms.RandomGrayscale(p=0.2)

        self.GlobalTransform1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=GlobalScale),
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.RandomHorizontalFlip(p=0.5),
            ColorJitter,
            GrayScale,
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            Normalize
        ])

        self.GlobalTransform2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=GlobalScale),
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.RandomHorizontalFlip(p=0.5),
            ColorJitter,
            GrayScale,
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.1),
            transforms.RandomSolarize(threshold=128, p=0.2),
            transforms.ToTensor(),
            Normalize
        ])

        self.NoLocalCrops = NoLocalCrops
        self.LocalTransform = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=LocalScale),   # FIX: was missing scale=
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.RandomHorizontalFlip(p=0.5),
            ColorJitter,
            GrayScale,
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
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
