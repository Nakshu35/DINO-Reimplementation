import torch
from tqdm.auto import tqdm

def Train(Student, Teacher, Loss, Optimizer, Device, TrainSet, Config):
    Student.train()
    Teacher.eval()
    TotalLoss = 0
    ProgressBar = tqdm(TrainSet, desc= "Training", leave= False)

    for images, _ in ProgressBar:
    
        images = [image.to(Device) for image in images]
    
        Studentfeatures = Student(images)
        Teacherfeatures = Teacher(images[:2])
    
        Studentfeatures = torch.concat(Studentfeatures, dim=0)
        Teacherfeatures = torch.concat(Teacherfeatures, dim=0)
    
        LossValue = Loss(Studentfeatures, Teacherfeatures)
        Optimizer.zero_grad()
        LossValue.backward()
        Optimizer.step()

        TotalLoss += LossValue.item()
    
        with torch.no_grad():
            for StudentParam, TeacherParam in zip(Student.parameters(), Teacher.parameters()):
                TeacherParam.data.mul_(Config["Momentum"])
                TeacherParam.data.add_((1 - Config["Momentum"]) * StudentParam.data)

        ProgressBar.set_postfix(
            loss=f"{LossValue.item():.4f}"
        )

    AvgLoss = TotalLoss / len(TrainSet.dataset)

    return AvgLoss, Optimizer.param_groups[0]["lr"]
