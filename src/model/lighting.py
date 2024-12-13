import lightning as L
import torchmetrics
from sklearn.metrics import f1_score
import torch


criterion = torch.nn.CrossEntropyLoss()

class LitBasic(L.LightningModule):
    def __init__(self, model):
      super().__init__()
      self.model = model
      self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=4)
      self.valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes=4)
      self.training_step_outputs = []
      self.training_step_targets = []
      self.val_step_outputs = []
      self.val_step_targets = []

    def configure_optimizers(self):
      optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
      return optimizer

    def training_step(self, batch, batch_idx):
      x, y = batch
      y_hat = self.model(x)
      train_loss = criterion(y_hat, y)

      self.log('train_loss', train_loss, on_step=False, on_epoch=True, prog_bar=True )
      y_pred = y_hat.argmax(dim=1).cpu().numpy()
      y_true = y.cpu().numpy()
      self.training_step_outputs.extend(y_pred)
      self.training_step_targets.extend(y_true)

      return train_loss

    def on_train_epoch_end(self):
      train_all_outputs = self.training_step_outputs
      train_all_targets = self.training_step_targets
      f1_macro_epoch = f1_score(train_all_outputs, train_all_targets, average='macro')
      self.log("training_f1_epoch", f1_macro_epoch, on_step=False, on_epoch=True, prog_bar=True)

      self.training_step_outputs.clear()
      self.training_step_targets.clear()


    def validation_step(self, batch, batch_idx):
      x, y = batch
      y_hat = self.model(x)
      val_loss = criterion(y_hat, y)

      self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True )

      y_pred = y_hat.argmax(dim=1).cpu().numpy()
      y_true = y.cpu().numpy()

      self.val_step_outputs.extend(y_pred)
      self.val_step_targets.extend(y_true)

      return val_loss

    def on_validation_epoch_end(self):
      val_all_outputs = self.val_step_outputs
      val_all_targets = self.val_step_targets
      val_f1_macro_epoch = f1_score(val_all_outputs, val_all_targets, average='macro')
      self.log("val_f1_epoch", val_f1_macro_epoch, on_step=False, on_epoch=True, prog_bar=True)
      self.val_step_outputs.clear()
      self.val_step_targets.clear()