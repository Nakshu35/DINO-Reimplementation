from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image

IndextoClasses = {
    0: "Tench",
    1: "English Springer",
    2: "Cassette Player",
    3: "Chain Saw",
    4: "Church",
    5: "French Horn",
    6: "Garbage Truck",
    7: "Gas Pump",
    8: "Golf Ball",
    9: "Parachute",
}

ClasstoIndex = {
    "n01440764": 0,  # Tench
    "n02102040": 1,  # English Springer
    "n02979186": 2,  # Cassette Player
    "n03000684": 3,  # Chain Saw
    "n03028079": 4,  # Church
    "n03394916": 5,  # French Horn
    "n03417042": 6,  # Garbage Truck
    "n03425413": 7,  # Gas Pump
    "n03445777": 8,  # Golf Ball
    "n03888257": 9,  # Parachute
}

SupervisedTransform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])

class Imagenette(Dataset):
    def __init__(self, DatasetDir=None, SetType='train', transform = SupervisedTransform):
        super().__init__()

        self.transform = transform

        SetTypeDir = os.path.join(DatasetDir,"DATA FILES",SetType)
        ImagesDir = []
        for f in os.listdir(SetTypeDir):
            for i in os.listdir(os.path.join(SetTypeDir, f)):
                ImagesDir.append((os.path.join(SetTypeDir, f, i), ClasstoIndex[f]))
        self.ImagesDir = ImagesDir


    def __len__(self):
        return len(self.ImagesDir)

    def __getitem__(self, Index):
        ImagePath, target = self.ImagesDir[Index]
        PathtoImage = Image.open(ImagePath)
        image = self.transform(PathtoImage)
        return image, target
