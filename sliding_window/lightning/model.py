# this program will create a class called SDGTagger for BERT fine-tuning model using pytorch lightning

from cProfile import label
from optparse import OptionError
import torch
import torch.nn as nn

# from transformers import BertTokenizerFast as AutoModelForSequenceClassification, AutoModel, AdamW, get_linear_schedule_with_warmup
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
import pytorch_lightning as pl
from torchmetrics import AUROC

from config import DEFAULT_PARAMS_BERT_WITH_POOLING, DEFAULT_PARAMS_DATA
N_CLASSES = len(DEFAULT_PARAMS_DATA['label_columns'])

class SDGTagger(pl.LightningModule):
    def __init__(self, n_classes: int = N_CLASSES, learning_rate=DEFAULT_PARAMS_BERT_WITH_POOLING['learning_rate'], n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.current_device = DEFAULT_PARAMS_BERT_WITH_POOLING['device']
        config = AutoConfig.from_pretrained(DEFAULT_PARAMS_BERT_WITH_POOLING['bert_model'])
        config.num_labels = n_classes
        config.problem_type = "multi_label_classification"
        self.bert_classifier = AutoModelForSequenceClassification.from_config(config).to(self.current_device)
        self.criterion = nn.BCEWithLogitsLoss()
        
        # self.bert = AutoModel.from_pretrained(DEFAULT_PARAMS_BERT_WITH_POOLING['bert_model'])
        # self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        # self.criterion = nn.BCELoss(reduction='mean')
        
        # self.bert = BertForSequenceClassifcation.from_pretrained(DEFAULT_PARAMS_BERT_WITH_POOLING['bert_model'], problem_type="multi_label_classification", )
        self.lr = learning_rate
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps

        self.pooling_method = DEFAULT_PARAMS_BERT_WITH_POOLING['pooling_strategy']

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert_classifier(input_ids, attention_mask=attention_mask)[0]
        return output

    def __evaluate_batch(self, batch, step):
        input_ids = batch[0]
        attention_masks = batch[1]
        labels = batch[2]
        
        total_outputs = []
        for sample_input, sample_attention in zip(input_ids, attention_masks):
            outputs = self.bert_classifier(input_ids=sample_input, attention_mask=sample_attention)[0]
            # pooling
            if self.pooling_method == 'max':
                total_outputs.append(torch.max(outputs, dim=0, keepdim=True)[0][0])

        labels = torch.stack(labels)
        total_outputs = torch.stack(total_outputs)

        loss = self.criterion(total_outputs, labels)
        
        self.log(f"{step}_loss", loss, prog_bar=True, logger=True)
        
        return {"loss": loss, "predictions": total_outputs.detach(), "labels": labels}


    
    def __evaluate_step(self, batch, step):
        input_ids = batch[0]
        attention_mask = batch[1]
        labels = batch[2]
        number_of_chunks = [len(x) for x in input_ids]

        # concatenate all input_ids into one batch
        input_ids_combined = []
        for x in input_ids:
            input_ids_combined.extend(x.tolist())

        input_ids_combined_tensors = torch.stack([torch.tensor(x).to(self.current_device) for x in input_ids_combined])

        # concatenate all attention masks into one batch
        attention_mask_combined = []
        for x in attention_mask:
            attention_mask_combined.extend(x.tolist())

        attention_mask_combined_tensors = torch.stack([torch.tensor(x).to(self.current_device) for x in attention_mask_combined])

        # get model predictions for the combined batch
        preds = self(input_ids_combined_tensors, attention_mask_combined_tensors)

        preds_split = preds.split(number_of_chunks)

        # pooling
        pooling_method = DEFAULT_PARAMS_BERT_WITH_POOLING['pooling_strategy']
        if pooling_method == 'max':
            pooled_preds = torch.stack([torch.max(chunk, dim=0, keepdim=True)[0][0] for chunk in preds_split])

        elif pooling_method == 'mean':
            pooled_preds = torch.stack([torch.mean(chunk, dim=0) for chunk in preds_split])

        labels = torch.stack(labels)
        loss = self.criterion(pooled_preds, labels)
        
        self.log(f"{step}_loss", loss, prog_bar=True, logger=True)
        
        return {"loss": loss, "predictions": pooled_preds, "labels": labels}

    def training_step(self, batch, batch_idx):
        # return self.__evaluate_batch(batch, "train")
        return self.__evaluate_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        # return self.__evaluate_batch(batch, "val")['loss']
        return self.__evaluate_step(batch, "val")['loss']

    def test_step(self, batch, batch_idx):
        # return self.__evaluate_batch(batch, "test")['loss']
        return self.__evaluate_step(batch, "test")['loss']

    def training_epoch_end(self, outputs):
        labels = []
        predictions = []
        for output in outputs:
            labels.extend(iter(output["labels"].detach().to(self.current_device)))
            predictions.extend(iter(output["predictions"].detach().to(self.current_device)))
        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)
        
        auroc = AUROC(len(DEFAULT_PARAMS_DATA['label_columns']))
        result = auroc(predictions, labels)
        self.logger.experiment.add_scalar("val_auroc/Train", result, self.current_epoch)

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        return {
          'val_loss': avg_loss,
          'progress_bar':{'val_loss': avg_loss}}

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)

        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=self.n_warmup_steps,
        #     num_training_steps=self.n_training_steps
        # )

        return dict(
            optimizer=optimizer
        )

        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='step'
            )
        )