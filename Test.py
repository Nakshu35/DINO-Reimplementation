import os
import json
import yaml
import torch
import torch.nn as nn
import argparse
from DATA.Dataset import Imagenette
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm 

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda image: image.convert("RGB")),
    transforms.ToTensor()
])


def ParseArgs():
    Parse = argparse.ArgumentParser()
    Parse.add_argument("--Config", type=str, required=True)
    return Parse.parse_args()


def ExtractEmbeddings(Backbone, Loader, Device):
    AllFeats, AllLabels = [], []
    with torch.no_grad():
        for images, labels in Loader:
            images = images.to(Device)
            features = Backbone(images)
            AllFeats.append(features.cpu())
            AllLabels.append(labels)
    return torch.cat(AllFeats, dim=0), torch.cat(AllLabels, dim=0)


def RunKNN(TrainFeats, TrainLabels, ValFeats, ValLabels, k=20):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(TrainFeats.numpy(), TrainLabels.numpy())
    return knn.score(ValFeats.numpy(), ValLabels.numpy())


def RunLinearProbe(TrainFeats, TrainLabels, ValFeats, ValLabels, Device, NumClasses, Epochs=50, LR=1e-3):
    LinearHead = nn.Linear(TrainFeats.shape[1], NumClasses).to(Device)
    Optimizer = torch.optim.AdamW(LinearHead.parameters(), lr=LR)
    Criterion = nn.CrossEntropyLoss()

    TrainFeats_d = TrainFeats.to(Device)
    TrainLabels_d = TrainLabels.to(Device)
    ProgressBar = tqdm(range(Epochs), desc="Linear probe training")

    LinearHead.train()
    for epoch in ProgressBar:
        Optimizer.zero_grad()
        Logits = LinearHead(TrainFeats_d)
        Loss = Criterion(Logits, TrainLabels_d)
        Loss.backward()
        Optimizer.step()
        ProgressBar.set_postfix(loss=Loss.item())

    LinearHead.eval()
    with torch.no_grad():
        ValLogits = LinearHead(ValFeats.to(Device))
        Preds = ValLogits.argmax(dim=1).cpu()
        Accuracy = (Preds == ValLabels).float().mean().item()

    return Accuracy


def main():
    Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Args = ParseArgs()
    with open(Args.Config) as f:
        Config = yaml.safe_load(f)

    ModelDir = os.path.join(Config["RootDir"], "Saves", Config["ExperimentName"], "Teacher.pt")
    Model = torch.load(ModelDir, weights_only=False)
    Backbone = Model.ViTBackbone
    Backbone.to(Device)
    Backbone.eval()

    TrainSet = Imagenette(RootDir=Config["DatasetDir"], SetType="train", transform=transform)
    ValSet   = Imagenette(RootDir=Config["DatasetDir"], SetType="val",   transform=transform)

    TrainLoader = DataLoader(dataset=TrainSet, batch_size=Config["BatchSize"], shuffle=False)
    ValLoader   = DataLoader(dataset=ValSet, batch_size=Config["BatchSize"], shuffle=False)

    TrainFeats, TrainLabels = ExtractEmbeddings(Backbone, TrainLoader, Device)
    ValFeats, ValLabels = ExtractEmbeddings(Backbone, ValLoader, Device)

    KnnAcc = RunKNN(TrainFeats, TrainLabels, ValFeats, ValLabels, k=20)

    LinProbeAcc = RunLinearProbe(
        TrainFeats, TrainLabels, ValFeats, ValLabels,
        Device=Device, NumClasses=Config["NumClasses"]
    )

    print(f"kNN accuracy: {KnnAcc:.4f}")
    print(f"Linear probe accuracy: {LinProbeAcc:.4f}")

    Results = {
        "experiment": Config["ExperimentName"],
        "knn_accuracy": KnnAcc,
        "linear_probe_accuracy": LinProbeAcc,
    }

    SaveDir = os.path.join(Config["RootDir"], "Saves", Config["ExperimentName"])
    os.makedirs(SaveDir, exist_ok=True)
    ResultsPath = os.path.join(SaveDir, f"{Config['ExperimentName']}_results.json")

    with open(ResultsPath, "w") as f:
        json.dump(Results, f, indent=2)

    print(f"Results saved to {ResultsPath}")


if __name__ == '__main__':
    main()
