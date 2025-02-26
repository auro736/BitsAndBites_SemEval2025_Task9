import os
import json
import pickle
import argparse
import numpy as np
from datetime import datetime

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from utils.utils import *
from utils.dataset import TextClassificationDataset
from utils.model import *

from transformers import AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup

torch.manual_seed(42)

def get_parser():

    parser = argparse.ArgumentParser(description="Training Configuration Parser")

    parser.add_argument("--save_model", type=bool, default=True,
                        help="Whether to save the trained model as pickle file (default: True)")
    parser.add_argument("--model_name", type=str, default="FacebookAI/roberta-large",
                        help="Pretrained model name or path (default: 'FacebookAI/roberta-large')")
    parser.add_argument("--num_epochs", type=int, default=8,
                        help="Number of training epochs (default: 8)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training (default: 8)")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate for optimization (default: 1e-5)")
    parser.add_argument("--lambda_h", type=float, default=0.5,
                        help="Lambda coefficient for hazard loss (default: 0.5)")
    parser.add_argument("--lambda_p", type=float, default=0.5,
                        help="Lambda coefficient for product loss (default: 0.5)")
    parser.add_argument("--mode", type=str, default="MH", choices=["MH", "MH+CN"],
                        help="Data modality (default: 'MH', options: 'MH', 'MH+CN')")
    parser.add_argument("--df_train_path", type=str, default="../data/internal_splits/train_internal.csv",
                        help="Path to the training dataset (default: '../data/internal_splits/train_internal.csv')")
    parser.add_argument("--df_validation_path", type=str, default="../data/internal_splits/val_internal.csv",
                        help="Path to the validation dataset (default: '../data/internal_splits/val_internal.csv')")
    parser.add_argument("--df_test_path", type=str, default="../data/internal_splits/test_internal.csv",
                        help="Path to the test dataset (default: '../data/internal_splits/test_internal.csv')")

    return parser

def main():
    
    # task 1 hazard
    # task 2 product

    parser = get_parser()
    args = parser.parse_args()

    print(f"Save Model: {args.save_model}")
    print(f"Model Name: {args.model_name}")
    print(f"Number of Epochs: {args.num_epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Lambda H: {args.lambda_h}")
    print(f"Lambda P: {args.lambda_p}")
    print(f"Mode: {args.mode}")
    print(f"Training Data Path: {args.df_train_path}")
    print(f"Validation Data Path: {args.df_validation_path}")
    print(f"Test Data Path: {args.df_test_path}")

    save_model = args.save_model
    model_name = args.model_name 
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    lambda_h = args.lambda_h
    lambda_p = args.lambda_p
    mode = args.mode
    df_train_path = args.df_train_path
    df_validation_path = args.df_validation_path
    df_test_path = args.df_test_path

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    save_model_dir = f"./ST2/saved_models/{model_name}_{timestamp}"
    os.makedirs(save_model_dir, exist_ok=True)
   
    performance_file = f"./ST2/performance/performance.txt"
    os.makedirs(os.path.dirname(performance_file), exist_ok=True)

    num_labels_hazard = 128
    num_labels_product = 1142

    performance_dict = {}

    performance_dict['model'] = model_name
    performance_dict['n_epochs'] = num_epochs
    performance_dict['learning_rate'] = learning_rate
    performance_dict['batch_size'] = batch_size
    performance_dict['lambda_h'] = lambda_h
    performance_dict['lambda_p'] = lambda_p
    performance_dict['saved_model_path'] = save_model_dir

    # Initialize tokenizer and config

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels_task1 = num_labels_hazard
    config.num_labels_task2 = num_labels_product

    # prepare data for training, validation and testing
    train_texts, val_texts, test_texts, \
    train_labels_hazard, val_labels_hazard, test_labels_hazard, \
    train_labels_product, val_labels_product, test_labels_product = prepare_data_st2(df_train_path, df_validation_path, df_test_path, mode)


    # initialize datasets and dataloaders

    train_dataset = TextClassificationDataset(train_texts, train_labels_hazard, train_labels_product, tokenizer, max_length=512)
    val_dataset = TextClassificationDataset(val_texts, val_labels_hazard, val_labels_product, tokenizer, max_length=512)
    test_dataset = TextClassificationDataset(test_texts, test_labels_hazard, test_labels_product, tokenizer, max_length=512)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)


    # Initialize the model    
    model = ModelForJointSequenceClassification(model_name,config)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Run training
    avg_train_loss, train_score, avg_val_loss, val_score = train(  model=model, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            num_epochs=num_epochs, 
            device=device, 
            lambda_h=lambda_h, 
            lambda_p=lambda_p, 
            output_dir=save_model_dir, 
            save_model = save_model,
            performance_dict=performance_dict
        )

    performance_dict['avg_train_loss'] = avg_train_loss
    performance_dict['train_score'] = train_score
    performance_dict['avg_val_loss'] = avg_val_loss
    performance_dict['val_score'] = val_score

    if save_model:
        best_model = pickle.load(open(save_model_dir+'/model.pkl', 'rb'))
        best_model.to(device)

        # Run testing
        test_score = test( model=best_model, 
                test_loader=test_loader, 
                device=device
            )
    else:
         test_score = test( model=model, 
                test_loader=test_loader, 
                device=device
            )

    performance_dict['test_score'] = test_score

    for key, value in performance_dict.items():
        if type(value) is np.int64:
            performance_dict[key] = int(value)
            
    with open(performance_file, 'a+') as outfile:
        outfile.write(json.dumps(performance_dict) + '\n')

    print(performance_dict)
    

if __name__ == '__main__':
    main()