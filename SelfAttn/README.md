
# Self-Attn 

## Project Overview
This project implements the training and inference stages of the Self-Attn as described in the paper **"A Fusion Analytic Framework for Investigating Functional Brain Connectivity Differences using resting state fMRI."** 

### Files:
- `generate_dummy_dataset.py`: Generates a random dataset with features in the form of (116, 116) matrices and binary labels.
- `self_attn.py`: Defines Self-Attn using TensorFlow Keras.
- `train.py`: Trains the model with 10-fold cross-validation.
- `inference.py`: Extracts the attention distribution matrix through the inference stage.

### Configuration:
All constants and variables are managed via the `config.yaml` file.

### Dependencies:
To install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
```

### How to Run:
1. Generate the dataset:
   ```bash
   python data/dataset.py
   ```
2. Train the model:
   ```bash
   python training/train.py
   ```
3. Run inference:
   ```bash
   python inference/inference.py
   ```

### Configuration File:
The `config.yaml` file contains important settings such as model architecture, training parameters, and file paths. Modify this file to adjust the hyperparameters and other configurations.

### Outputs:
- The trained model checkpoints will be saved in the `outputs/checkpoints/` directory.
- The final attention distribution matrix and labels will be saved in the `outputs/` directory after running the inference.
