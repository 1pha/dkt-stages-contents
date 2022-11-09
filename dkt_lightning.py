import os

import torch
import torchmetrics
import pytorch_lightning as pl

import sys
sys.path.append("./")

from code.dkt.src.trainer import process_batch, compute_loss, get_model

class DktLightning(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.model = get_model(args)

        self.accuracy = torchmetrics.Accuracy()
        self.auroc = torchmetrics.AUROC(num_classes=3)

    def configure_optimizers(self):
        # TODO: hard-coded optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, train_batch, batch_idx):

        input = process_batch(train_batch)
        preds = self.forward(input)
        targets = input[3]
        loss = compute_loss(preds, targets)

        preds = preds[:, -1]
        targets = targets[:, -1]
        self.accuracy(preds, targets.int())
        self.auroc(preds, targets.int())

        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", self.accuracy)
        self.log("train_auroc", self.auroc)
        return loss

    def validation_step(self, validate_batch, batch_idx):

        input = process_batch(validate_batch)
        preds = self.forward(input)
        targets = input[3]
        loss = compute_loss(preds, targets)

        preds = preds[:, -1]
        targets = targets[:, -1]
        self.accuracy(preds, targets.int())
        self.auroc(preds, targets.int())

        self.log("validate_loss", loss)
        self.log("validate_acc", self.accuracy)
        self.log("validate_auroc", self.auroc)
        return loss

    def predict_step(self, predict_batch, batch_idx):

        input = process_batch(predict_batch)
        preds = self.forward(input)
        preds = torch.nn.Sigmoid()(preds[:, :-1])
        return preds[:, -1].cpu().detach()

    def on_predict_epoch_end(self, results):
        write_path = os.path.join(self.args.output_dir, "submission.csv")
        total_preds = torch.cat(results[0]).tolist()
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)
        with open(write_path, "w", encoding="utf8") as w:
            w.write("id,prediction\n")
            for id, p in enumerate(total_preds):
                w.write("{},{}\n".format(id, p))
        return total_preds
