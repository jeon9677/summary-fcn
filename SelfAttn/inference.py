import numpy as np
import tensorflow as tf
import yaml

# Load configuration
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Load data
X = np.load('data/X.npy')
y = np.load('data/y.npy')

# Initialize final attention distribution matrix for all samples
final_attention_distribution_matrix = np.zeros((X.shape[0], *config['data']['input_shape']))

# Inference and attention extraction
for i in range(X.shape[0]):

    # Initialize attention distribution matrix for each sample
    sample_attention_distribution = np.zeros(config['data']['input_shape'])

    # For each fold, compute the attention weights
    for fold in range(config['data']['k_fold']):

        # Load the model for the current fold
        model = tf.keras.models.load_model(f'outputs/checkpoints/model_fold_{fold + 1}.keras', compile=False)

        # Extract the MultiHeadAttention layer from the model
        attention_layer = model.get_layer(index=1)  # Assuming MultiHeadAttention is the second layer

        # Expand dims to match input shape (batch_size, seq_len, features)
        sample = np.expand_dims(X[i], axis=0)

        # Forward pass through the model with return_attention_scores=True
        _, attention_weights = attention_layer(sample, sample, return_attention_scores=True)

        # Average attention weights across all heads for this fold
        mean_attention_scores = np.mean(attention_weights, axis=1)  # Shape: (batch_size, query_len, key_len)

        # Add the attention scores for this fold to the sample's attention distribution
        sample_attention_distribution += mean_attention_scores[0]  # Remove batch dimension (query_len, key_len)

    # Average across all folds for this sample
    sample_attention_distribution /= config['data']['k_fold']

    # Save the averaged attention distribution for this sample
    final_attention_distribution_matrix[i] = sample_attention_distribution

# Save the final attention distribution matrix and labels
np.save('outputs/attention_distribution_matrix.npy', final_attention_distribution_matrix)

print("Inference complete. Attention distribution matrix saved.")
