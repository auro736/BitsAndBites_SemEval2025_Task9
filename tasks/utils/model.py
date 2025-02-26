from transformers import PreTrainedModel, AutoModel

import torch.nn as nn
from torch.nn import CrossEntropyLoss

class ModelForJointSequenceClassification(PreTrainedModel):  
    authorized_unexpected_keys = [r"pooler"]
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, model_name, config):
        super().__init__(config)

        self.num_labels_task1 = config.num_labels_task1
        self.num_labels_task2 = config.num_labels_task2
        self.model = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)

        # Due teste di classificazione
        self.classifier_task1 = nn.Linear(config.hidden_size, self.num_labels_task1)
        self.classifier_task2 = nn.Linear(config.hidden_size, self.num_labels_task2)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels_task1=None,
            labels_task2=None,
            return_dict=None,
            lambda_h=None,
            lambda_p =None,
            **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask
        )

        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0, :]  

        pooled_output = self.dropout(pooled_output)

        # Both head logits 
        logits_task1 = self.classifier_task1(pooled_output)
        logits_task2 = self.classifier_task2(pooled_output)

        loss = None

        # If one of the two labels are not provided only one classification is performed
        if labels_task1 is not None:
            task1_loss_fct = CrossEntropyLoss()
            loss_task1 = task1_loss_fct(logits_task1.view(-1, self.num_labels_task1), labels_task1.view(-1))
        if labels_task2 is not None:
            task2_loss_fct = CrossEntropyLoss()
            loss_task2 = task2_loss_fct(logits_task2.view(-1, self.num_labels_task2), labels_task2.view(-1))

        # Loss combination
        if labels_task1 is not None and labels_task2 is not None:
            loss =  lambda_h * loss_task1 + lambda_p *  loss_task2
            
        elif labels_task1 is not None:
            loss = loss_task1
        elif labels_task2 is not None:
            loss = loss_task2

        output = (logits_task1, logits_task2) + outputs[2:]
        return ((loss,) + output) if loss is not None else output

