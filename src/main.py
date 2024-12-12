from pathlib import Path
import torch
from data.dataset import create_dataset
from model.model import CNN, train_loop, validate
from torch.utils.data import DataLoader
from torchsummary import summary

dataset = create_dataset(Path(r'data\united\cfs'), Path(r'data\united\perm'))
train_loader = DataLoader(dataset[0], batch_size=3)
test_loader = DataLoader(dataset[1], batch_size=1)

device = 'cpu'
model = CNN().to(device)

# print(summary(model, (1, 150, 12)))
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

num_epochs = 10

for i in range(num_epochs):
  train_loss =  train_loop(
      train_loader,
      model,
      criterion,
      optimizer,
      device
  )
  print(train_loss)

accuracy = validate(model, test_loader, device)

print(f"Accuracy on TEST {accuracy:.2f}")