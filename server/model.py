import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class ImageClassification(nn.Module):
    def __init__(self):
        super(ImageClassification, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1, stride=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 512, 1024)
        self.fc2 = nn.Linear(1024, 2)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.pool(self.bn4(F.relu(self.conv4(x))))
        x = x.view(-1, 7 * 7 * 512)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def preprocess(data_dir):
    batch_size = 8

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.4919, 0.4615, 0.4179], [0.2557, 0.2470, 0.2503])
    ])

    image_dataset = datasets.ImageFolder(os.path.join(data_dir),
                                            data_transforms)

    dataloader = DataLoader(image_dataset, 
                            batch_size=batch_size, 
                            shuffle=True)

    return (dataloader, len(image_dataset))

def evaluation(model, data):
    since = time.time()
    model.eval()
    total_predict_true = 0
    for inputs, labels in data[0]:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        total_predict_true += len(torch.where(preds == labels)[0])
    return (total_predict_true / data[1]) * 100

if __name__ == "__main__":
    dataloader = preprocess("/home/thebreaker/Downloads/cat-and-dog/val")
    model = torch.load("server/checkpoints/model.pth")
    acc = evaluation(model, dataloader)    
    print("The accuracy for this model is ", acc)