import argus
import timm
import torch
import torch.nn as nn
import torch.optim as optim


class BCEMSELoss(nn.Module):
    def __init__(self, bce_weight=1.0, mse_weight=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.mse_weight = mse_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    def forward(self, input, target):
        logits, student_features, teacher_features = input

        if self.bce_weight > 0:
            bce = self.bce(logits, target) * self.bce_weight
        else:
            bce = 0

        if self.mse_weight > 0:
            mse = self.mse(student_features, teacher_features) * self.mse_weight
        else:
            mse = 0

        return bce + mse

    
class StudentModel(nn.Module):
    def __init__(self, teacher_model, **params):
        super().__init__()
        num_classes = params['num_classes']
        student_model = timm.create_model(**params)
        if params['model_name'] in ['rexnet_150']:
            n_features = student_model.head.fc.in_features
            student_model.head.global_pool = nn.Identity()
            student_model.head.fc = nn.Identity()
        elif params['model_name'] in ['resnet200d', 'resnet50d']:
            n_features = student_model.fc.in_features
            student_model.global_pool = nn.Identity()
            student_model.fc = nn.Identity()
        else:
            n_features = student_model.classifier.in_features
            student_model.global_pool = nn.Identity()
            student_model.classifier = nn.Identity()

        self.student_model = student_model
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_features, num_classes)

        self.teacher_model = teacher_model

    def forward(self, input):
        orig_input, input = input
        bz = input.size(0)

        with torch.no_grad():
            teacher_features = self.teacher_model.forward_features(input)

        student_features = self.student_model(orig_input)
        logits = self.fc(self.pooling(student_features).view(bz, -1))

        return logits, student_features, teacher_features


class RANZCRStageZero(argus.Model):
    nn_module = timm.create_model
    optimizer = optim.AdamW
    loss = nn.BCEWithLogitsLoss
    prediction_transform = nn.Sigmoid

    
class RANZCRStageOne(argus.Model):
    nn_module = StudentModel
    optimizer = optim.AdamW
    loss = BCEMSELoss

    class CustomSigmoid(nn.Sigmoid):
        def forward(self, input):
            return super().forward(input[0])

    prediction_transform = CustomSigmoid
