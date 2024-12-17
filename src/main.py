from pathlib import Path
import torch
from data.dataset import create_dataset
from model.model import  DNN, train_loop, validate
from torch.utils.data import DataLoader
from torchsummary import summary

dataset = create_dataset(Path(r'data\united\cfs'), Path(r'data\united\perm'))
train_loader = DataLoader(dataset[0], batch_size=5)
test_loader = DataLoader(dataset[1], batch_size=1)

device = 'cpu'
model = DNN().to(device)
model.train(True)
# print(summary(model, (4, 3, 150)))  
criterion = torch.nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=1e-5)

num_epochs = 100

for i in range(num_epochs):
  train_loss =  train_loop(
      train_loader,
      model,
      criterion,
      optimizer,
      device
  )
  print(train_loss)
