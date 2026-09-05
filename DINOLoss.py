import torch.nn as nn
import torch

class DINOLoss(nn.Module):
    def __init__(self, NCrops, TeacherTemp = 0.04 , StudentTemp = 0.1, CenterMomentum = 0.9, OutputDim= 65536):
        super().__init__()

        self.NCrops = NCrops
        self.StudentTemp = StudentTemp
        self.TeacherTemp = TeacherTemp
        self.CenterMomentum = CenterMomentum
        self.register_buffer("center", torch.zeros(1, OutputDim))

    def forward(self, StudentOutput, TeacherOutput):
        StudentOutput /= self.StudentTemp
        StudentOutput = StudentOutput.chunk(self.NCrops)

        with torch.no_grad():
            batch_center = TeacherOutput.mean(dim=0, keepdim=True)

            self.center.mul_(self.CenterMomentum)
            self.center.add_(batch_center * (1 - self.CenterMomentum))

        TeacherOutput = torch.softmax((TeacherOutput - self.CenterMomentum) / self.TeacherTemp, dim=-1)
        TeacherOutput = TeacherOutput.detach().chunk(2)

        TotalLoss = 0.0
        NLossTerms = 0

        for ID, Value in enumerate(TeacherOutput):
            for StudentID in range(len(StudentOutput)):
                if StudentID == ID:
                    continue
                Loss = torch.sum(-Value * torch.log_softmax(StudentOutput[StudentID], dim=-1), dim=-1)
                TotalLoss += Loss.mean()
                NLossTerms += 1

        TotalLoss /= NLossTerms

        return TotalLoss


