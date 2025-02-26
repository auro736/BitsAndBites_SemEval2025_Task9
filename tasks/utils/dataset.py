import torch
from torch.utils.data import Dataset

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels_task1, labels_task2, tokenizer, max_length=128):

        self.texts = texts
        self.labels_task1 = labels_task1
        self.labels_task2 = labels_task2
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        
        if self.labels_task1 is not None:
            item['labels_task1'] = torch.tensor(self.labels_task1[idx], dtype=torch.long)
        if self.labels_task2 is not None:
            item['labels_task2'] = torch.tensor(self.labels_task2[idx], dtype=torch.long)
        return item
    


class BaselineTextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.float) 
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], 
                                  add_special_tokens = True, 
                                  padding="max_length", 
                                  truncation=True, 
                                  max_length=self.max_length, 
                                  return_tensors="pt",
                                  return_attention_mask=True,
                                  )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": self.labels[idx] 
            }