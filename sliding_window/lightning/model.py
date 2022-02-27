# this program will create a class called SDGTagger for BERT fine-tuning model using pytorch lightning

import torch
import torch.nn as nn

from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

import pytorch_lightning as pl
from torchmetrics import AUROC

import numpy as np

from config import BERT_MODEL_NAME, DEFAULT_PARAMS_DATA
N_CLASSES = len(DEFAULT_PARAMS_DATA['label_columns'])

from transformers import logging
logging.set_verbosity_warning()

import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

class SDGTagger(pl.LightningModule):
    def __init__(self, n_classes: int = N_CLASSES, n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = nn.BCELoss(reduction='sum')

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)
        loss = self.criterion(output, labels) if labels is not None else 0
        return loss, output

    def __evaluate_step(self, batch, step):
        
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        number_of_chunks = [len(x) for x in input_ids]

        # concatenate all input_ids into one batch
        input_ids_combined = []
        for x in input_ids:
            input_ids_combined.extend(x.tolist())

        input_ids_combined_tensors = torch.stack([torch.tensor(x).to('cpu') for x in input_ids_combined])

        # concatenate all attention masks into one batch
        attention_mask_combined = []
        for x in attention_mask:
            attention_mask_combined.extend(x.tolist())

        attention_mask_combined_tensors = torch.stack([torch.tensor(x).to('cpu') for x in attention_mask_combined])


        # get model predictions for the combined batch
        loss , preds = self(input_ids_combined_tensors, attention_mask_combined_tensors)

        preds_split = preds.split(number_of_chunks)

        # pooling
        pooling_method = "mean"
        if pooling_method == 'max':
            pooled_preds = torch.stack([torch.max(x, dim=0) for x in preds_split])

        elif pooling_method == 'mean':
            pooled_preds = torch.stack([torch.mean(chunk, dim=0) for chunk in preds_split])

        loss = self.criterion(pooled_preds, labels)
        
        self.log(f"{step}_loss", loss, prog_bar=True, logger=True)
        
        return {"loss": loss, "predictions": pooled_preds, "labels": labels}

    def training_step(self, batch, batch_idx):
        return self.__evaluate_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.__evaluate_step(batch, "val")['loss']

    def test_step(self, batch, batch_idx):
        return self.__evaluate_step(batch, "test")['loss']

    def training_epoch_end(self, outputs):
        labels = []
        predictions = []
        for output in outputs:
            labels.extend(iter(output["labels"].detach().cpu()))
            predictions.extend(iter(output["predictions"].detach().cpu()))
        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)
        
        auroc = AUROC(len(DEFAULT_PARAMS_DATA['label_columns']))
        result = auroc(predictions, labels)
        self.logger.experiment.add_scalar("val_auroc/Train", result, self.current_epoch)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps
        )

        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='step'
            )
        )