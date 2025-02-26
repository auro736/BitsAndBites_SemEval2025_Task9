import os
import ast
import torch
import pickle
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score




def prepare_data(df_path_train, df_path_validation, df_path_test, mode):
    
    df_train = pd.read_csv(df_path_train, index_col = 0)
    df_validation = pd.read_csv(df_path_validation, index_col = 0)
    df_test = pd.read_csv(df_path_test, index_col = 0)
    
    if mode == 'MH':
        train_texts = df_train['full_text'].tolist()
        val_texts = df_validation['full_text'].tolist()
        test_texts = df_test['full_text'].tolist()
    elif mode == 'MH+CN':
        train_texts = df_train['enriched_texts'].tolist()
        val_texts = df_validation['enriched_texts'].tolist()
        test_texts = df_test['enriched_texts'].tolist()

    train_labels_hazard_cat = df_train['hazard-cat-label'].tolist()
    val_labels_hazard_cat = df_validation['hazard-cat-label'].tolist()
    test_labels_hazard_cat = df_test['hazard-cat-label'].tolist()

    train_labels_product_cat = df_train['product-cat-label'].tolist()
    val_labels_product_cat = df_validation['product-cat-label'].tolist()
    test_labels_product_cat = df_test['product-cat-label'].tolist()

    return train_texts, val_texts, test_texts, train_labels_hazard_cat, val_labels_hazard_cat, test_labels_hazard_cat, train_labels_product_cat, val_labels_product_cat, test_labels_product_cat


def prepare_data_st2(df_path_train, df_path_validation, df_path_test, mode):
    
    df_train = pd.read_csv(df_path_train, index_col = 0)
    df_validation = pd.read_csv(df_path_validation, index_col = 0)
    df_test = pd.read_csv(df_path_test, index_col = 0)

    if mode == 'MH':
        train_texts = df_train['full_text'].tolist()
        val_texts = df_validation['full_text'].tolist()
        test_texts = df_test['full_text'].tolist()
    elif mode == 'MH+CN':
        train_texts = df_train['enriched_texts'].tolist()
        val_texts = df_validation['enriched_texts'].tolist()
        test_texts = df_test['enriched_texts'].tolist()
        
    train_labels_hazard = df_train['hazard-label'].tolist()
    val_labels_hazard = df_validation['hazard-label'].tolist()
    test_labels_hazard = df_test['hazard-label'].tolist()

    train_labels_product = df_train['product-label'].tolist()
    val_labels_product = df_validation['product-label'].tolist()
    test_labels_product = df_test['product-label'].tolist()

    return train_texts, val_texts, test_texts, train_labels_hazard, val_labels_hazard, test_labels_hazard, train_labels_product, val_labels_product, test_labels_product


def compute_score(hazards_true, products_true, hazards_pred, products_pred):
  # compute f1 for hazards:
  f1_hazards = f1_score(
    hazards_true,
    hazards_pred,
    average='macro'
  )

  # compute f1 for products:
  f1_products = f1_score(
    products_true[hazards_pred == hazards_true],
    products_pred[hazards_pred == hazards_true],
    average='macro'
  )

  return (f1_hazards + f1_products) / 2

def train(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device, output_dir, lambda_h, lambda_p, save_model, performance_dict):

    best_val_score = 0  
    print('Epochs to run: ', num_epochs)
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        hazards_true_train = []
        products_true_train = []
        hazards_pred_train = []
        products_pred_train = []

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels_task1=batch['labels_task1'],
                labels_task2=batch['labels_task2'],
                lambda_h=lambda_h,
                lambda_p=lambda_p
            )
            loss = outputs[0]
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            scheduler.step()

            # Collect predictions and ground truth for hazards and products
            logits_task1 = outputs[1]
            logits_task2 = outputs[2]

            probs_task_1 = torch.nn.functional.softmax(logits_task1, dim=1)
            probs_task_2 = torch.nn.functional.softmax(logits_task2, dim=1)

            preds_task1 = torch.argmax(probs_task_1, dim=1)
            preds_task2 = torch.argmax(probs_task_2, dim=1)

            hazards_true_train.append(batch['labels_task1'])
            products_true_train.append(batch['labels_task2'])
            hazards_pred_train.append(preds_task1)
            products_pred_train.append(preds_task2)

        avg_train_loss = total_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss}")

        # Concatenate predictions and true labels for training
        hazards_true_train = torch.cat(hazards_true_train).cpu()
        products_true_train = torch.cat(products_true_train).cpu()
        hazards_pred_train = torch.cat(hazards_pred_train).cpu()
        products_pred_train = torch.cat(products_pred_train).cpu()

        # Compute F1 score for training
        train_score = compute_score(
            hazards_true_train.numpy(),
            products_true_train.numpy(),
            hazards_pred_train.numpy(),
            products_pred_train.numpy()
        )
        print(f"Training F1 Challenge Score: {train_score}")

        # Validation
        model.eval()
        val_loss = 0
        hazards_true_val = []
        products_true_val = []
        hazards_pred_val = []
        products_pred_val = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels_task1=batch['labels_task1'],
                    labels_task2=batch['labels_task2'],
                    lambda_h=lambda_h,
                    lambda_p=lambda_p
                )
                loss = outputs[0]
                val_loss += loss.item()

                logits_task1 = outputs[1]
                logits_task2 = outputs[2]

                probs_task_1 = torch.nn.functional.softmax(logits_task1, dim=1)
                probs_task_2 = torch.nn.functional.softmax(logits_task2, dim=1)
                
                preds_task1 = torch.argmax(probs_task_1, dim=1)
                preds_task2 = torch.argmax(probs_task_2, dim=1)

                hazards_true_val.append(batch['labels_task1'])
                products_true_val.append(batch['labels_task2'])
                hazards_pred_val.append(preds_task1)
                products_pred_val.append(preds_task2)

        avg_val_loss = val_loss / len(val_loader)

        # Concatenate predictions and true labels for validation
        hazards_true_val = torch.cat(hazards_true_val).cpu()
        products_true_val = torch.cat(products_true_val).cpu()
        hazards_pred_val = torch.cat(hazards_pred_val).cpu()
        products_pred_val = torch.cat(products_pred_val).cpu()

        # Compute F1 score for validation
        val_score = compute_score(
            hazards_true_val.numpy(),
            products_true_val.numpy(),
            hazards_pred_val.numpy(),
            products_pred_val.numpy()
        )
        print(f"Validation Loss: {avg_val_loss}")
        print(f"Validation F1 Challenge Score: {val_score}")

        # Save model if validation F1 score improved
        if save_model:
            if val_score > best_val_score:
                best_val_score = val_score
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                # Save the model using pickle
                pickle.dump(model, open(os.path.join(output_dir, 'model.pkl'), 'wb'))
                print("Best model saved.")
                performance_dict['epoch_best_model'] = epoch+1
        
    return avg_train_loss, train_score, avg_val_loss, best_val_score if best_val_score != 0 else val_score


def test(model, test_loader, device):
    model.eval()
    hazards_true = []
    products_true = []
    hazards_pred = []
    products_pred = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
            )
            logits_task1 = outputs[0]
            logits_task2 = outputs[1]

            probs_task_1 = torch.nn.functional.softmax(logits_task1, dim=1)
            probs_task_2 = torch.nn.functional.softmax(logits_task2, dim=1)
            
            preds_task1 = torch.argmax(probs_task_1, dim=1)
            preds_task2 = torch.argmax(probs_task_2, dim=1)

            hazards_true.append(batch['labels_task1'])
            products_true.append(batch['labels_task2'])
            hazards_pred.append(preds_task1)
            products_pred.append(preds_task2)
    
    hazards_true = torch.cat(hazards_true).cpu()
    products_true = torch.cat(products_true).cpu()
    hazards_pred = torch.cat(hazards_pred).cpu()
    products_pred = torch.cat(products_pred).cpu()

    test_score = compute_score(
        hazards_true.numpy(),
        products_true.numpy(),
        hazards_pred.numpy(),
        products_pred.numpy()
    )

    print(f"Test F1 Challenge Score: {test_score}")

    return test_score


def predict(model, test_loader, device, label_task1_mapping, label_task2_mapping):
   
    model.eval()
    predictions = []

    #task 1 hazard
    #task 2 product

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
            )
            logits_task1 = outputs[0]
            logits_task2 = outputs[1]

            probs_task_1 = torch.nn.functional.softmax(logits_task1, dim=1)
            probs_task_2 = torch.nn.functional.softmax(logits_task2, dim=1)

            preds_task1 = torch.argmax(probs_task_1, dim=1)
            preds_task2 = torch.argmax(probs_task_2, dim=1)
            
            preds_task1 = [label_task1_mapping[i.item()] for i in preds_task1]
            preds_task2 = [label_task2_mapping[i.item()] for i in preds_task2]
            
            predictions.extend(zip(preds_task1, preds_task2))
    
    return predictions

def SDC_masking(category_model, detail_model, test_loader, device, det_label_task1_mapping, det_label_task2_mapping, category_to_detail_map_task_1, category_to_detail_map_task_2):
    
    category_model.eval()
    detail_model.eval()
    predictions = []

    #task 1 hazard
    #task 2 product

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Double Predicting"):
            batch = {k: v.to(device) for k, v in batch.items()}

            category_pred = category_model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
            )

            cat_logits_task1 = category_pred[0] # logits di task 1 aka hazard category
            cat_logits_task2 = category_pred[1] # logits di task 1 aka product category

            cat_probs_task_1 = torch.nn.functional.softmax(cat_logits_task1, dim=1)
            cat_probs_task_2 = torch.nn.functional.softmax(cat_logits_task2, dim=1)

            cat_preds_task1 = torch.argmax(cat_probs_task_1, dim=1) # categoria predetta di hazard
            cat_preds_task2 = torch.argmax(cat_probs_task_2, dim=1) # categoria predetta di product

            detail_pred = detail_model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
            )

            det_logits_task1 = detail_pred[0] # logits di task 1 aka hazard 
            det_logits_task2 = detail_pred[1] # logits di task 2 aka product 
            
            batch_size = cat_preds_task1.size(0)
            num_det_task_1 = det_logits_task1.size(1)
            num_det_task_2 = det_logits_task2.size(1)
            masks_task_1 = []
            masks_task_2 = []

            for i in range(batch_size):
                m_1 = torch.zeros(num_det_task_1, device=device)
                m_2 = torch.zeros(num_det_task_2, device=device)

                pred_cat_task_1 = cat_preds_task1[i].item()
                pred_cat_task_2 = cat_preds_task2[i].item()

                valid_details_1 = category_to_detail_map_task_1.get(pred_cat_task_1, [])
                valid_details_2 = category_to_detail_map_task_2.get(pred_cat_task_2, [])
             
                m_1[valid_details_1] = 1
                m_2[valid_details_2] = 1

                masks_task_1.append(m_1)
                masks_task_2.append(m_2)

            mask_1_tensor = torch.stack(masks_task_1)
            mask_2_tensor = torch.stack(masks_task_2)
            
            masked_detail_logits_1 = det_logits_task1.clone()
            masked_detail_logits_1[mask_1_tensor == 0] = float('-inf')

            masked_detail_logits_2 = det_logits_task2.clone()
            masked_detail_logits_2[mask_2_tensor == 0] = float('-inf')
            
            # Otteniamo le predizioni finali dei prodotti
            probs_masked_detail_1 = torch.nn.functional.softmax(masked_detail_logits_1, dim=1)
            probs_masked_detail_2 = torch.nn.functional.softmax(masked_detail_logits_2, dim=1)

            detail_preds_1 = torch.argmax(probs_masked_detail_1, dim=1)
            detail_preds_2 = torch.argmax(probs_masked_detail_2, dim=1)

            detail_preds_1_categorico = [det_label_task1_mapping[i.item()] for i in detail_preds_1]
            detail_preds_2_categorico = [det_label_task2_mapping[i.item()] for i in detail_preds_2]
   
            predictions.extend(zip(detail_preds_1_categorico, detail_preds_2_categorico))
    
    return predictions

def SDC_probs_multiplication(category_model, detail_model, test_loader, device, det_label_task1_mapping, det_label_task2_mapping, category_to_detail_map_task_1, category_to_detail_map_task_2):
    
    category_model.eval()
    detail_model.eval()
    predictions = []

    #task 1 hazard
    #task 2 product

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Double Predicting"):
            batch = {k: v.to(device) for k, v in batch.items()}

            category_pred = category_model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
            )

            cat_logits_task1 = category_pred[0] # logits di task 1 aka hazard category
            cat_logits_task2 = category_pred[1] # logits di task 1 aka product category

            cat_probs_task_1 = torch.nn.functional.softmax(cat_logits_task1, dim=1)
            cat_probs_task_2 = torch.nn.functional.softmax(cat_logits_task2, dim=1)


            detail_pred = detail_model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
            )

            det_logits_task1 = detail_pred[0] # logits di task 1 aka hazard 
            det_logits_task2 = detail_pred[1] # logits di task 2 aka product 

            det_probs_task_1 = torch.nn.functional.softmax(det_logits_task1, dim=1)
            det_probs_task_2 = torch.nn.functional.softmax(det_logits_task2, dim=1)

            det_probs_task_1_modified = torch.zeros_like(det_probs_task_1)
            det_probs_task_2_modified = torch.zeros_like(det_probs_task_2)

            for idx_categoria in range(cat_probs_task_1.shape[1]):  
                if idx_categoria in category_to_detail_map_task_1:
                    indices = category_to_detail_map_task_1[idx_categoria]  
                    det_probs_task_1_modified[:, indices] += det_probs_task_1[:, indices] * cat_probs_task_1[:, idx_categoria].unsqueeze(1)

            for idx_categoria in range(cat_probs_task_2.shape[1]): 
                if idx_categoria in category_to_detail_map_task_2:
                    indices = category_to_detail_map_task_2[idx_categoria]
                    det_probs_task_2_modified[:, indices] += det_probs_task_2[:, indices] * cat_probs_task_2[:, idx_categoria].unsqueeze(1)

            detail_preds_1 = torch.argmax(det_probs_task_1_modified, dim=1)
            detail_preds_2 = torch.argmax(det_probs_task_2_modified, dim=1)

            detail_preds_1_categorico = [det_label_task1_mapping[i.item()] for i in detail_preds_1]
            detail_preds_2_categorico = [det_label_task2_mapping[i.item()] for i in detail_preds_2]
   
            predictions.extend(zip(detail_preds_1_categorico, detail_preds_2_categorico))
    
    return predictions
    
def generate_predictions_csv(model, test_loader, device, task, label_task1_mapping, label_task2_mapping, output_csv_path, columns):
    if task == 'ST1':
        label_task1_mapping = {1: 'biological',
                            4: 'foreign bodies',
                            2: 'chemical',
                            5: 'fraud',
                            7: 'organoleptic aspects',
                            0: 'allergens',
                            9: 'packaging defect',
                            8: 'other hazard',
                            3: 'food additives and flavourings',
                            6: 'migration'}
        
        label_task2_mapping = {   13: 'meat, egg and dairy products',
                                18: 'prepared dishes and snacks',
                                1: 'cereals and bakery products',
                                3: 'confectionery',
                                12: 'ices and desserts',
                                0: 'alcoholic beverages',
                                9: 'fruits and vegetables',
                                16: 'other food product / mixed',
                                2: 'cocoa and cocoa preparations, coffee and tea',
                                15: 'nuts, nut products and seeds',
                                19: 'seafood',
                                20: 'soups, broths, sauces and condiments',
                                5: 'fats and oils',
                                14: 'non-alcoholic beverages',
                                8: 'food contact materials',
                                4: 'dietetic foods, food supplements, fortified foods',
                                10: 'herbs and spices',
                                7: 'food additives and flavourings',
                                21: 'sugars and syrups',
                                11: 'honey and royal jelly',
                                6: 'feed materials',
                                17: 'pet feed'}
        
    elif task == 'ST2':
        with open('../data/mappings/id2label_hazard.txt', 'r') as file:
            data_h = file.read()
        label_task1_mapping = ast.literal_eval(data_h)

        with open('../data/mappings/id2label_product.txt', 'r') as file:
            data_p = file.read()
        label_task2_mapping = ast.literal_eval(data_p)

    predictions = predict(model, test_loader, device, label_task1_mapping, label_task2_mapping)
    
    df = pd.DataFrame(predictions, columns=columns)
    
    df.to_csv(output_csv_path)
    print(f"Predictions saved to {output_csv_path}")

    return df