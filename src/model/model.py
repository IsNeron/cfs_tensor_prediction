import torch
import torch.nn as nn
from torch.nn.modules.pooling import MaxPool2d



def get_correct_count(pred, labels):
    _, predicted = torch.max(pred.data, 1)
    return (predicted.cpu() == labels.cpu()).sum().item()



@torch.inference_mode()
def validate(model, test_loader, device="cpu"):
    correct, total = 0, 0
    for imgs, labels in test_loader:
        pred = model(imgs.to(device))
        total += labels.size(0)
        correct += get_correct_count(pred, labels)
    return correct / total



class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers_stack = nn.Sequential(
            nn.Linear(4*3*150,512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 9),
      )
        self.nn_layers = nn.ModuleList()

    def forward(self, x):
        x = self.flatten(x)
        x = self.layers_stack(x)
        x = x.reshape(-1, 3, 3)
        return x



def train_loop(
        dataloader, 
        model, 
        criterion, 
        optimizer, 
        device
):
    num_batches = len(dataloader)

    train_loss = 0

    for cfs, labels in dataloader:
        pred =  model(cfs.to(device))
        loss =  criterion(pred, labels.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # print('PRED ', pred)
        # print('LABELS ', labels)

    train_loss /= num_batches

    return train_loss