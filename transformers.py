from tensorflow import keras


class TransformerEncoderBlock(keras.layers.Layer):
    def __init__(self, head_size, num_heads, ff_dim, dropout=0, activation='relu', **kwargs):
        super().__init__(**kwargs)
        # Normalization and Attention
        self.mha_norm = keras.layers.LayerNormalization(epsilon=1e-6)
        self.mha = keras.layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout, name='multi-head-attention'
        )
        self.mha_dropout = keras.layers.Dropout(dropout)

        # Feed Forward Part
        self.ff_norm = keras.layers.LayerNormalization(epsilon=1e-6)
        self.ff1 = keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation=activation)
        self.ff_dropout = keras.layers.Dropout(dropout)


    def build(self, input_shape):
        super().build(input_shape)
        self.ff2 = keras.layers.Conv1D(filters=input_shape[-1], kernel_size=1)


    def call(self, inputs, training=None):
        x = self.mha_norm(inputs, training=training)
        x = self.mha(x, x, training=training)
        x = self.mha_dropout(x, training=training)
        res = x + inputs

        x = self.ff_norm(res, training=training)
        x = self.ff1(x, training=training)
        x = self.ff_dropout(x, training=training)
        x = self.ff2(x, training=training)
        return x + res


    def get_config(self):
        config = super().get_config()
        config.update({'head_size': self.mha.key_dim})
        config.update({'num_heads': self.mha.num_heads})
        config.update({'ff_dim': self.ff1.filters})
        config.update({'dropout': self.mha_dropout.dropout})
        config.update({'activation': self.ff1.activation})
        return config
