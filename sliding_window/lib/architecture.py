import torch.nn as nn


class BERTSequenceClassificationHead(nn.Module):
    def __init__(self, num_labels = 1):

        super().__init__()

        # dropout layer
        #self.dropout = nn.Dropout(0.1)

        # dense layer 1
        #self.dense = nn.Linear(768,768)

        # dense layer 2 (Output layer)
        self.out_proj = nn.Linear(768, num_labels)

        # sigmoid activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, cls_token_hidden_state):
        x = cls_token_hidden_state
        #x = self.dropout(x)
        #x = self.dense(x)
        #x = torch.tanh(x)
        #x = self.dropout(x)
        x = self.out_proj(x)

        # apply softmax activation
        x = self.sigmoid(x)

        return x


class BERTSequenceClassificationArch(nn.Module):

    def __init__(self, bert, num_labels=1):

        super().__init__()

        self.bert = bert
        self.classification_head = BERTSequenceClassificationHead(num_labels)

    # define the forward pass
    def forward(self, input_ids, attention_mask):

        # pass the inputs to the model
        outputs = self.bert(input_ids, attention_mask)
        sequence_output = outputs[0]

        x = sequence_output[:, 0, :]  # take <s> token (equiv. to [CLS])

        # pass vectorized output to classification head
        x = self.classification_head(x)
        return x