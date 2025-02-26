import os
import torch
import pickle
import argparse
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments

from utils.utils import compute_score
from utils.dataset import BaselineTextClassificationDataset


def get_parser():

    parser = argparse.ArgumentParser(description="Baseline training configuration")

    parser.add_argument("--model", type=str, default="roberta-large",
                        help="Model name or path (default: 'roberta-large')")
    parser.add_argument("--save_model", type=bool, default=True,
                        help="Whether to save the trained model as pickle file (default: True)")
    parser.add_argument("--epochs", type=int, default=8,
                        help="Number of training epochs (default: 8)")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate for optimization (default: 1e-5)")
    parser.add_argument("--save_strategy", type=str, default="no", choices=["no", "epoch", "steps"],
                        help="Strategy for saving the model (default: 'no', options: 'no', 'epoch', 'steps')")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training (default: 8)")
    
    return parser

def main():

    parser = get_parser()
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Save Model: {args.save_model}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print(f"Save Strategy: {args.save_strategy}")
    print(f"Batch Size: {args.batch_size}")

    save_model = args.save_model
    epochs =args.epochs
    lr = args.lr
    save_strategy = args.save_strategy
    batch_size = args.batch_size

    output_dir = "./ST1/out_baseline/saved_models"

    true_validation = pd.read_csv('../data/raw/incidents_valid.csv', index_col=0)
    true_test = pd.read_csv('../data/raw/incidents_test.csv', index_col = 0)

    train_df = pd.read_csv('../data/corpus_normalization/output/cn_incidents_train.csv', index_col=0)
    valid_df = pd.read_csv('../data/corpus_normalization/output/cn_incidents_validation.csv', index_col = 0)
    test_df = pd.read_csv('../data/corpus_normalization/output/cn_incidents_test.csv', index_col=0)

    train_df['text'] = train_df['text'].str.replace('\n', '')
    train_df['text'] = train_df['text'].str.replace(r'\s+', ' ', regex=True)

    valid_df['text'] = valid_df['text'].str.replace('\n', '')
    valid_df['text'] = valid_df['text'].str.replace(r'\s+', ' ', regex=True)

    valid_df['full_text'] = valid_df['title'] + valid_df['text']
    valid_texts = valid_df['full_text'].tolist()

    test_df['full_text'] = test_df['title'] + test_df['text']
    test_texts = test_df['full_text'].tolist()

    cat_labels = train_df[['hazard-category', 'product-category']].values.tolist()

    mlb_categories = MultiLabelBinarizer()
    y_train = mlb_categories.fit_transform(cat_labels)

    num_labels = len(mlb_categories.classes_) 

    tokenizer = RobertaTokenizer.from_pretrained(args.model)

    train_df['full_text'] = train_df['title'] + train_df['text']
    texts = train_df['full_text'].tolist()

    labels = y_train.tolist()

    train_dataset = BaselineTextClassificationDataset(texts=texts, labels=labels, tokenizer=tokenizer, max_length=512)

    model = RobertaForSequenceClassification.from_pretrained(args.model, 
                                                             num_labels=num_labels, 
                                                             problem_type = "multi_label_classification")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(model.config.problem_type)



    training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    eval_strategy="no",
    save_strategy=save_strategy,
    logging_dir="./ST1/out_baseline/logs",
    learning_rate = lr,
    save_only_model=True,
    seed = 42
    )

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
    )

    trainer.train()

    if save_model:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Save the model using pickle
        pickle.dump(model, open(os.path.join(output_dir, 'model.pkl'), 'wb'))
        print("Model saved.")

    # Predizione su valid e test
    preds_validation = predict_and_save(device, model, tokenizer, valid_texts, "./ST1/out_baseline/predictions/st1_baseline_preds_valid.csv", mlb_categories)
    preds_test = predict_and_save(device, model, tokenizer, test_texts, "./ST1/out_baseline/predictions/st1_baseline_preds_test.csv", mlb_categories)

    score_validation = compute_score(
            true_validation['hazard-category'].to_numpy(), 
            true_validation['product-category'].to_numpy(), 
            preds_validation['hazard-category'].to_numpy(), 
            preds_validation['product-category'].to_numpy()
            )
    
    score_test = compute_score(
            true_test['hazard-category'].to_numpy(), 
            true_test['product-category'].to_numpy(), 
            preds_test['hazard-category'].to_numpy(), 
            preds_test['product-category'].to_numpy()
            )
    
    print(f"After {epochs} epochs: \n")
    print(f"Score on the validations set: {round(score_validation, 4)} \n")
    print(f"Score on the test set: {round(score_test, 4)} \n")


def predict_and_save(device, model, tokenizer, predict_texts, output_csv, mlb_categories, max_length=512):
    
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for text in predict_texts:
            encoding = tokenizer(
                text,
                add_special_tokens=True,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            top2_indices = probs.argsort()[-2:][::-1]  
            labels = mlb_categories.classes_[top2_indices]
            predictions.append(labels)
    
    pred_df = pd.DataFrame(predictions, columns=["hazard-category", "product-category"])
    pred_df.to_csv(output_csv)
    print(f"Predictions saved to {output_csv}")

    return pred_df






if __name__ == '__main__':
    main()