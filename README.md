# BitAndBites - SemEval 2025 Task 9

This repository is dedicated to participating in **SemEval 2025 Task 9**, which focuses on **food hazard prediction**.

## Task Overview
SemEval-Task 9 combines two sub-tasks:

1. **Text Classification for Food Hazard Prediction (ST1)**  
   - **Objective**: Predict the type of hazard and the associated product from the text.

2. **Food Hazard and Product "Vector" Detection (ST2)**  
   - **Objective**: Predict the exact hazard and product mentioned in the text.

---

## Repository Structure

### **Utils**
- `tasks/utils/` → Contains utility scripts:
  - `model.py` → Defines multi-head model class.
  - `dataset.py` → Contains dataset processing classes.
  - `utils.py` → Contains helper functions for training and inference.

### **Baseline Models**
- `tasks/baseline_ST1.py` → Training script for **ST1** using a single-head classifier (concatenating title and text).
- `taksk/baseline_ST2.py` → Training script for **ST2** using a single-head classifier (concatenating title and text).
  - **Arguments:**
    - `--model` → Model name or path (default: `'roberta-large'`).
    - `--save_model` → Whether to save the trained model as a pickle file (default: `True`).
    - `--epochs` → Number of training epochs (default: `8`).
    - `--lr` → Learning rate for optimization (default: `1e-5`).
    - `--save_strategy` → Strategy for saving the model (default: `'no'`, options: `'no'`, `'epoch'`, `'steps'`).
    - `--batch_size` → Batch size for training (default: `8`).

### **Multi-Head Models**
- `tasks/main_ST1.py` → Training script for **ST1** using a multi-head classifier.
- `tasks/main_ST2.py` → Training script for **ST2** using a multi-head classifier.
  - **Arguments:**
    - `--save_model` → Whether to save the trained model as a pickle file (default: `True`).
    - `--model_name` → Pretrained model name or path (default: `'FacebookAI/roberta-large'`).
    - `--num_epochs` → Number of training epochs (default: `8`).
    - `--batch_size` → Batch size for training (default: `8`).
    - `--learning_rate` → Learning rate for optimization (default: `1e-5`).
    - `--lambda_h` → Lambda coefficient for hazard loss (default: `0.5`).
    - `--lambda_p` → Lambda coefficient for product loss (default: `0.5`).
    - `--mode` → Data modality (default: `'MH'`, options: `'MH' (without Corpus Normalization)`, `'MH+CN' (with Corpus Normalization)`).
    - `--df_train_path` → Path to the training dataset (default: `'../data/internal_splits/train_internal.csv'`).
    - `--df_validation_path` → Path to the validation dataset (default: `'../data/internal_splits/val_internal.csv'`).
    - `--df_test_path` → Path to the test dataset (default: `'../data/internal_splits/test_internal.csv'`).

### **Sequence Classification (SC) Inference for ST2**
- `tasks/SDC_inference_ST2.py` → Inference script for **ST2** using **sequence double classification (SDC)**.
  - **Arguments:**
    - `--saved_ST1_model_dir` → Path to the saved ST1 model pickle file (default: `''`).
    - `--saved_ST2_model_dir` → Path to the saved ST2 model pickle file (default: `''`).
    - `--model_name` → Saved model name to load tokenizer (default: `'FacebookAI/roberta-large'`).
    - `--batch_size` → Batch size for evaluation (default: `8`).
    - `--mode` → Data modality (default: `'MH'`, options: `'MH' (without Corpus Normalization)`, `'MH+CN' (with Corpus Normalization)`).
    - `--sc_technique` → SC technique to apply (default: `'probs_multiplication'`, options: `'masking'`, `'probs_multiplication'`).

---

## Example Commands

### **Train Baseline Models**
```bash
python baseline_ST1.py --epochs 10 --batch_size 32 --lr 5e-5
python baseline_ST2.py --epochs 10 --batch_size 32 --lr 5e-5
```

### **Train Multi-Head Models**
```bash
python main_ST1.py --mode 'MH' --num_epochs 8 --batch_size 8 --learning_rate 1e-5 --lambda_h 0.5 --lambda_p 0.5
python main_ST2.py --mode 'MH+CN' --num_epochs 8 --batch_size 8 --learning_rate 1e-5 --lambda_h 0.5 --lambda_p 0.5
```

### **Run SC Inference for ST2**
```bash
python SC_inference_ST2.py --saved_ST1_model_dir 'path/to/st1_model.pkl' --saved_ST2_model_dir 'path/to/st2_model.pkl' --mode 'MH+CN' --sc_technique 'masking'
python SC_inference_ST2.py --saved_ST1_model_dir 'path/to/st1_model.pkl' --saved_ST2_model_dir 'path/to/st2_model.pkl' --mode 'MH' --sc_technique 'probs_multiplication'
```

---

## Requirements
Install dependencies using:
```bash
pip install -r requirements.txt
```

---

## Contributors
BitAndBites Team - SemEval 2025 Task 9

---

## License
This project is released under the [MIT License](LICENSE).





