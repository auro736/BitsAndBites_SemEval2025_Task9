import ast
import json
import pickle

from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from utils.utils import *
from utils.dataset import TextClassificationDataset

import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="SDC in inference parser")

    parser.add_argument("--saved_ST1_model_dir", type=str, required = True,
                        help="Path to the saved ST1 model pickle file (default: '')")
    parser.add_argument("--saved_ST2_model_dir", type=str, required = True,
                        help="Path to the saved ST1 model pickle file (default: '')")
    parser.add_argument("--model_name", type=str, default="FacebookAI/roberta-large",
                        help="Saved model name to load tokenizer (default: 'FacebookAI/roberta-large')")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for evaluation (default: 8)")
    parser.add_argument("--mode", type=str, default="MH", choices=["MH", "MH+CN"],
                        help="Data modality (default: 'MH', options: 'MH', 'MH+CN')")
    parser.add_argument("--sc_tecnique", type=str, default="probs_multiplication", choices=["masking", "probs_multiplication"],
                        help="SDC technique to apply (default: 'probs_multiplication', options: 'masking', 'probs_multiplication')")

    return parser


def main():

    parser = get_parser()
    args = parser.parse_args()

    valid_preds_output_dir = f"./ST2/predictions/sc_{args.sc_tecnique}_{args.mode}_preds_on_valid.csv"
    test_preds_output_dir = f"./ST2/predictions/sc_{args.sc_tecnique}_{args.mode}_preds_on_test.csv"

    print(f"ST1 model directory: {args.saved_ST1_model_dir}")
    print(f"ST2 model directory: {args.saved_ST2_model_dir}")
    print(f"Model Name: {args.model_name}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Data modality: {args.mode}")
    print(f"SDC Technique: {args.sc_tecnique}")
    print(f"Validation Predictions Output Path: {valid_preds_output_dir}")
    print(f"Test Predictions Output Path: {test_preds_output_dir}")

    saved_ST1_model_dir = args.saved_ST1_model_dir
    saved_ST2_model_dir = args.saved_ST2_model_dir
    model_name = args.model_name
    batch_size = args.batch_size
    mode = args.mode
    sc_tecnique = args.sc_tecnique
    
    true_validation = pd.read_csv('../data/raw/incidents_valid.csv', index_col=0)
    true_test = pd.read_csv('../data/raw/incidents_test.csv', index_col = 0)

    validation_df_path = '../data/corpus_normalization/output/cn_incidents_validation.csv' 
    test_df_path = '../data/corpus_normalization/output/cn_incidents_test.csv' 
   
    df_validation = pd.read_csv(validation_df_path, index_col = 0)
    df_test = pd.read_csv(test_df_path, index_col = 0)

    if mode == 'MH':
        df_validation['full_text'] = df_validation['title'] + " " + df_validation['text']
        validation_texts = df_validation['full_text'].tolist()

        df_test['full_text'] = df_test['title'] + " " + df_test['text']
        test_texts = df_test['full_text'].tolist()

    elif mode == 'MH+CN':
        validation_texts = df_validation['enriched_texts'].tolist()
        test_texts = df_test['enriched_texts'].tolist()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    validation_dataset = TextClassificationDataset(validation_texts, labels_task1=None, labels_task2=None, tokenizer=tokenizer, max_length=512)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)

    test_dataset = TextClassificationDataset(test_texts, labels_task1=None, labels_task2=None, tokenizer=tokenizer, max_length=512)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    category_model = pickle.load(open(saved_ST1_model_dir, 'rb'))
    detail_model = pickle.load(open(saved_ST2_model_dir, 'rb'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    category_model.to(device)
    detail_model.to(device)

    with open('../data/mappings/id2label_product.txt', 'r') as file:
        data_p = file.read()
    id_2_label_product = ast.literal_eval(data_p)

    with open('../data/mappings/id2label_hazard.txt', 'r') as file:
        data_h = file.read()
    id_2_label_hazard = ast.literal_eval(data_h)

    with open("../data/mappings/mapping_category_detail_product.json", "r") as json_file:
        mapping_category_detail_product = json.load(json_file)
    mapping_category_detail_product = {int(k): v for k, v in mapping_category_detail_product.items()}

    with open("../data/mappings/mapping_category_detail_hazard.json", "r") as json_file:
        mapping_category_detail_hazard = json.load(json_file)
    mapping_category_detail_hazard = {int(k): v for k, v in mapping_category_detail_hazard.items()}
    
    if sc_tecnique.lower() == 'masking':
        preds_validation = SDC_masking(category_model=category_model, 
                    detail_model=detail_model, 
                    test_loader=validation_loader, 
                    device=device, 
                    det_label_task1_mapping=id_2_label_hazard, 
                    det_label_task2_mapping=id_2_label_product, 
                    category_to_detail_map_task_1= mapping_category_detail_hazard, 
                    category_to_detail_map_task_2= mapping_category_detail_product
                    )
        
        preds_test = SDC_masking(category_model=category_model, 
                    detail_model=detail_model, 
                    test_loader=test_loader, 
                    device=device, 
                    det_label_task1_mapping=id_2_label_hazard, 
                    det_label_task2_mapping=id_2_label_product, 
                    category_to_detail_map_task_1= mapping_category_detail_hazard, 
                    category_to_detail_map_task_2= mapping_category_detail_product
                    )

    elif sc_tecnique.lower() == 'probs_multiplication':
        preds_validation = SDC_probs_multiplication(category_model=category_model, 
                    detail_model=detail_model, 
                    test_loader=validation_loader, 
                    device=device, 
                    det_label_task1_mapping=id_2_label_hazard, 
                    det_label_task2_mapping=id_2_label_product, 
                    category_to_detail_map_task_1= mapping_category_detail_hazard, 
                    category_to_detail_map_task_2= mapping_category_detail_product
                    )
        
        preds_test = SDC_probs_multiplication(category_model=category_model, 
                    detail_model=detail_model, 
                    test_loader=test_loader, 
                    device=device, 
                    det_label_task1_mapping=id_2_label_hazard, 
                    det_label_task2_mapping=id_2_label_product, 
                    category_to_detail_map_task_1= mapping_category_detail_hazard, 
                    category_to_detail_map_task_2= mapping_category_detail_product
                    )

    df_preds_valid = pd.DataFrame(preds_validation, columns=['hazard', 'product'])
    df_preds_valid.to_csv(valid_preds_output_dir)

    df_preds_test = pd.DataFrame(preds_test, columns=['hazard', 'product'])
    df_preds_test.to_csv(test_preds_output_dir)

    print(f"Predictions on validation set saved to {valid_preds_output_dir}")
    print(f"Predictions on test set saved to {test_preds_output_dir}")

    score_validation = compute_score(
            true_validation['hazard'].to_numpy(), 
            true_validation['product'].to_numpy(), 
            df_preds_valid['hazard'].to_numpy(), 
            df_preds_valid['product'].to_numpy()
            )
    
    score_test = compute_score(
            true_test['hazard'].to_numpy(), 
            true_test['product'].to_numpy(), 
            df_preds_test['hazard'].to_numpy(), 
            df_preds_test['product'].to_numpy()
            )
    
    print(f"With Sequence Classification (SC) tecnique: {sc_tecnique}")
    print(f"Score on the validations set: {round(score_validation, 4)} \n")
    print(f"Score on the test set: {round(score_test, 4)} \n")
    

if __name__ == '__main__':
    main()