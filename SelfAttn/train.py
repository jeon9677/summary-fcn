import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
import yaml
from self_attn import build_model

def one_hot_encode(y, num_classes=2):
    """
    Method to convert binary labels into one-hot format
    y: input labels (1 or 2)
    num_classes: total number of classes (default is 2)
    """
    # Adjust labels to match indices (1 => 0, 2 => 1)
    y_adjusted = y - 1
    return np.eye(num_classes)[y_adjusted]

# Load configuration
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Load dataset
X = np.load('data/X.npy')
y = np.load('data/y.npy')

# Cross-validation setup using StratifiedKFold
skf = StratifiedKFold(n_splits=config['data']['k_fold'], shuffle=True, random_state=config['random_seed'])

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold + 1}/{skf.n_splits}")

    # Split data
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    y_train_one_hot = one_hot_encode(y_train, num_classes=config['data']['num_classes'])
    y_test_one_hot = one_hot_encode(y_test, num_classes=config['data']['num_classes'])

    # Build model
    model = build_model(config['data']['input_shape'],
                        config['model']['num_heads'],
                        config['model']['key_dim'],
                        config['model']['dropout'])

    best_acc = 0

    # Training loop
    for epoch in range(config['training']['epochs']):
        history = model.fit(X_train, y_train_one_hot, batch_size=config['data']['batch_size'], epochs=1, verbose=0)

        # Evaluate
        test_loss, test_acc = model.evaluate(X_test, y_test_one_hot, verbose=0)

        # Save model if accuracy improves
        if test_acc > best_acc and epoch >= config['training']['min_epochs_before_save']:
            print(f"Epoch: {epoch} - Test accuracy improved to {test_acc}, saving model.")
            model.save(f'outputs/checkpoints/model_fold_{fold + 1}.keras')
            best_acc = test_acc
