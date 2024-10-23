import numpy as np
import yaml

# Load configuration
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Generate random dataset
def create_dataset(num_samples, input_shape, seed):
    np.random.seed(seed)
    X = np.random.rand(num_samples, *input_shape)

    # Use numpy to create random binary labels in the form of [1, 0] or [0, 1]
    y = np.random.choice([1, 2], num_samples)

    return X, y

if __name__ == "__main__":
    X, y = create_dataset(config['data']['num_samples'],
                          config['data']['input_shape'],
                          config['random_seed'])
    np.save('data/X.npy', X)
    np.save('data/y.npy', y)
