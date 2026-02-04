import torch
import torch.nn as nn

class FractureClassifier(nn.Module):
    def __init__(self):
        super(FractureClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)  # assuming input size 224x224
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # -> [B, 16, 112, 112]
        x = self.pool(self.relu(self.conv2(x)))  # -> [B, 32, 56, 56]
        x = x.view(x.size(0), -1)                # flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_model(model_path):
    model = FractureClassifier()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model
