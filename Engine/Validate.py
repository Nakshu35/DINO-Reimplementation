import torch
from tqdm.auto import tqdm

def Validate(Student, Teacher, ValidationSet, Loss, Device):
    Student.eval()
    Teacher.eval()
    TotalLoss = 0.0
    ProgressBar = tqdm(ValidationSet, desc="Validating", leave=False)

    for images, _ in ProgressBar:
        images = images.to(Device)

        StudentFeatures = Student(images)
        TeacherFeatures = Teacher(images)

        StudentFeatures = StudentFeatures[0]
        TeacherFeatures = TeacherFeatures[0]

        LossValue = Loss(StudentFeatures, TeacherFeatures)
        TotalLoss += LossValue.item()

        ProgressBar.set_postfix(
            loss=f"{LossValue.item():.4f}"
        )

    AvgLoss = TotalLoss / len(ValidationSet.dataset)
    
    return AvgLoss