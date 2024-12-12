import torch
import torch.nn as nn
from torch.nn.modules.pooling import MaxPool2d

def get_correct_count(pred, labels):
    _, predicted = torch.max(pred.data, 1)
    return (predicted.cpu() == labels.cpu()).sum().item()


@torch.inference_mode()  # this annotation disable grad computation
def validate(model, test_loader, device="cpu"):
    correct, total = 0, 0
    for imgs, labels in test_loader:
        pred = model(imgs.to(device))
        total += labels.size(0)
        correct += get_correct_count(pred, labels)
    return correct / total

#[(W−K+2P)/S]+1
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers_stack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=1, padding=1),
            nn.LeakyReLU(),
            MaxPool2d((4,1), stride=(2,1)),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=2, padding=2),
            nn.LeakyReLU(),
            MaxPool2d((4,1), stride=(2,1)),
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=3, padding=3),
            nn.LeakyReLU(),
            MaxPool2d((4,1), stride=(2,1)),
            nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            MaxPool2d((4,1), stride=(2,1)),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            MaxPool2d((4,19), stride=(2,1))
          
      )
        self.nn_layers = nn.ModuleList()

    def forward(self, x):
        scores = self.layers_stack(x)
        return scores

def train_loop(
        dataloader, 
        model, 
        criterion, 
        optimizer, 
        device
):
    num_batches = len(dataloader)

    train_loss = 0
    y_true, y_pred = torch.Tensor(), torch.Tensor()

    for imgs, labels in dataloader:
        pred =  model(imgs.to(device))
        loss =  criterion(pred, labels.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        y_true = torch.cat([y_true, labels], dim=0)
        pred_labels = pred.detach().cpu().argmax(dim=1)
        y_pred = torch.cat([y_pred, pred_labels], dim=0)

    train_loss /= num_batches

    return train_loss