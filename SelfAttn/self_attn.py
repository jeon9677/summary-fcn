import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention, Dense, Dropout, Flatten
import yaml

# Load configuration
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


# Define Self-Attention Model
def build_model(input_shape, num_heads, key_dim, dropout_rate):
    inputs = tf.keras.Input(shape=input_shape)

    # Multi-head self-attention layer
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout_rate)(inputs, inputs)

    # Flatten the output
    x = Flatten()(attn_output)

    # Output layer (binary classification)
    outputs = Dense(config['data']['num_classes'], activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate']),
                  loss=config['training']['loss'],
                  metrics=config['training']['metrics'])

    return model


if __name__ == "__main__":
    model = build_model(config['data']['input_shape'],
                        config['model']['num_heads'],
                        config['model']['key_dim'],
                        config['model']['dropout'])
    model.summary()
