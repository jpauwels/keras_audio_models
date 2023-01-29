from .transformers import TransformerEncoderBlock
from tensorflow import keras
from os.path import join, dirname
from collections.abc import Iterable 


def load_weights(model, weights):
    if weights:
        try:
            # First try built-in weights
            path = join(dirname(__file__), f'{weights}.h5')
            model.load_weights(path, by_name=True, skip_mismatch=True)
        except OSError:
            # Then check if path is HDF5 file containing weights only
            try:
                model.load_weights(f'{weights}.h5', by_name=True, skip_mismatch=True)
            # Otherwise the path points to a saved whole model
            except OSError:
                source_model = keras.models.load_model(weights, compile=False, custom_objects={'MusicnnFrontend': MusicnnFrontend, 'MusicnnMidend': MusicnnMidend, 'MusicnnBackend': MusicnnBackend})
                model.set_weights(source_model.get_weights())


def build_musicnn_classifier(
    inputs,
    num_classes=2,
    final_activation='sigmoid',
    num_filt_frontend=1.6,
    num_filt_midend=64,
    num_units_backend=(200, 100),
    backend_dropout=0.5,
    weights=None,
    **kwargs
):
    musicnn = build_musicnn(inputs, num_classes, num_filt_frontend, num_filt_midend, num_units_backend, backend_dropout, weights, **kwargs)
    musicnn.get_layer('backend').classifier.get_layer('logits').activation = keras.activations.get(final_activation)
    model = keras.Model(inputs=musicnn.input, outputs=musicnn.get_layer('backend').output, name='musicnn_classifier', **kwargs)
    return model


def build_musicnn(
    inputs,
    num_classes=50,
    num_filt_frontend=1.6,
    num_filt_midend=64,
    num_units_backend=200,
    backend_dropout=0.5,
    weights=None,
    **kwargs
):
    frontend_features_list = MusicnnFrontend(num_filt_frontend, name='frontend')(inputs)
    frontend_features = keras.layers.Concatenate(axis=-1, name='concat_frontend')(frontend_features_list)

    midend_features_list = MusicnnMidend(num_filt_midend, name='midend')(frontend_features)
    midend_features = keras.layers.Concatenate(axis=-1, name='concat_midend')(midend_features_list)

    logits = MusicnnBackend(num_classes, num_units_backend, backend_dropout, name='backend')(midend_features)
    taggram = keras.layers.Activation(keras.activations.sigmoid, name='taggram')(logits)

    model = keras.Model(inputs=inputs, outputs=taggram, name='musicnn', **kwargs)
    load_weights(model, weights)
    return model


def build_musicnn_transformer_classifier(
    inputs,
    num_classes=2,
    final_activation='sigmoid',
    num_filt_frontend=1.6,
    num_filt_midend=64,
    transformer_blocks=1,
    transformer_head_size=753,
    transformer_num_heads=1,
    num_units_transformer=200,
    transformer_dropout=0.1,
    num_units_backend=(200, 100),
    backend_dropout=0.5,
    weights=None,
    **kwargs
):
    musicnn = build_musicnn_transformer(inputs, num_classes, num_filt_frontend, num_filt_midend, transformer_blocks, transformer_head_size, transformer_num_heads, num_units_transformer, transformer_dropout, num_units_backend, backend_dropout, weights, **kwargs)
    musicnn.get_layer('backend').classifier.get_layer('logits').activation = keras.activations.get(final_activation)
    model = keras.Model(inputs=musicnn.input, outputs=musicnn.get_layer('backend').output, name='musicnn_transformer_classifier', **kwargs)
    return model


def build_musicnn_transformer(
    inputs,
    num_classes=50,
    num_filt_frontend=1.6,
    num_filt_midend=64,
    transformer_blocks=1,
    transformer_head_size=753,
    transformer_num_heads=1,
    num_units_transformer=100,
    transformer_dropout=0.1,
    num_units_backend=200,
    backend_dropout=0.5,
    weights=None,
    **kwargs
):
    frontend_features_list = MusicnnFrontend(num_filt_frontend, name='frontend')(inputs)
    frontend_features = keras.layers.Concatenate(axis=-1, name='concat_frontend')(frontend_features_list)

    midend_features_list = MusicnnMidend(num_filt_midend, name='midend')(frontend_features)
    x = keras.layers.Concatenate(axis=-1, name='concat_midend')(midend_features_list)

    for idx in range(transformer_blocks):
        x = TransformerEncoderBlock(transformer_head_size, transformer_num_heads, num_units_transformer, transformer_dropout, name=f'transformer{idx+1}')(x)

    logits = MusicnnBackend(num_classes, num_units_backend, backend_dropout, name='backend')(x)
    taggram = keras.layers.Activation(keras.activations.sigmoid, name='taggram')(logits)

    model = keras.Model(inputs=inputs, outputs=taggram, name='musicnn_transformer', **kwargs)
    load_weights(model, weights)
    return model


class Musicnn(keras.Model):
    def __init__(self, num_classes, num_filt_frontend=1.6, num_filt_midend=64, num_units_backend=200, **kwargs):
        super().__init__(**kwargs)

        self.frontend = MusicnnFrontend(num_filt_frontend, name='frontend')
        self.frontend_concat = keras.layers.Concatenate(axis=2, name='concat_frontend')

        self.midend = MusicnnMidend(num_filt_midend, name='midend')
        self.midend_concat = keras.layers.Concatenate(axis=2, name='concat_midend')

        self.backend = MusicnnBackend(num_classes, num_units_backend, name='backend')
        self.final_activation = keras.layers.Activation(keras.activations.sigmoid, name='taggram')


    def call(self, inputs, training=None):
        frontend_features_list = self.frontend(inputs, training=training)
        frontend_features = self.frontend_concat(frontend_features_list)

        midend_features_list = self.midend(frontend_features, training=training)
        midend_features = self.midend_concat(midend_features_list)

        logits = self.backend(midend_features, training=training)
        return self.final_activation(logits)


    def get_config(self):
        config = super().get_config()
        config.update({'num_classes': self.backend.get_config()['num_classes']})
        config.update({'num_filt_frontend': self.frontend.get_config()['num_filt']})
        config.update({'num_filt_midend': self.midend.get_config()['num_filt']})
        config.update({'num_units_backend': self.backend.get_config()['num_units']})
        return config


class MusicnnFrontend(keras.layers.Layer):
    def __init__(self, num_filt, **kwargs):
        super().__init__(**kwargs)
        self.num_filt = num_filt
        # hardcoded type '7774timbraltemporal'
        self.normalized_input = keras.layers.BatchNormalization(name='input_normalization')
        self.s1 = MusicnnFrontendBlock(filters=int(num_filt*32),
            kernel_size=(128, 1), padding="same", name='s1')
        self.s2 = MusicnnFrontendBlock(filters=int(num_filt*32),
            kernel_size=(64, 1), padding="same", name='s2')
        self.s3 = MusicnnFrontendBlock(filters=int(num_filt*32),
            kernel_size=(32, 1), padding="same", name='s3')


    def build(self, input_shape):
        super().build(input_shape)
        if len(input_shape) == 4:
            self.timbral_padding = keras.layers.ZeroPadding3D(padding=(0, 3, 0), name='timbral_padding')
        else:
            self.timbral_padding = keras.layers.ZeroPadding2D(padding=(3, 0), name='timbral_padding')
        self.f74 = MusicnnFrontendBlock(filters=int(self.num_filt*128),
            kernel_size=(7, int(0.4 * input_shape[-1])), padding="valid", name='f74')
        self.f77 = MusicnnFrontendBlock(filters=int(self.num_filt*128),
            kernel_size=(7, int(0.7 * input_shape[-1])), padding="valid", name='f77')


    def call(self, inputs, training=None):
        add_channel = keras.backend.expand_dims(inputs)
        normalized_input = self.normalized_input(add_channel, training=training)
        input_pad_7 =  self.timbral_padding(normalized_input)
        f74 = self.f74(input_pad_7, training=training)
        f77 = self.f77(input_pad_7, training=training)
        s1 = self.s1(normalized_input, training=training)
        s2 = self.s2(normalized_input, training=training)
        s3 = self.s3(normalized_input, training=training)
        return f74, f77, s1, s2, s3

    def get_config(self):
        config = super().get_config()
        config.update({'num_filt': self.num_filt})
        return config


class MusicnnFrontendBlock(keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding, activation=keras.activations.relu, **kwargs):
        super().__init__(**kwargs)
        self.conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
            padding=padding, activation=activation, name='conv')
        self.bn_conv = keras.layers.BatchNormalization(name='bn_conv')


    def build(self, input_shape):
        super().build(input_shape)
        if self.conv.padding == 'valid':
            conv_cols = input_shape[-2] - self.conv.kernel_size[1] + 1
        else:
            conv_cols = input_shape[-2]
        if len(input_shape) == 5:
            self.maxpool = keras.layers.MaxPooling3D(pool_size=(1, 1, conv_cols), name='maxpool')
        else:
            self.maxpool = keras.layers.MaxPooling2D(pool_size=(1, conv_cols), name='maxpool')


    def call(self, inputs, training=None):
        conv = self.conv(inputs)
        bn_conv = self.bn_conv(conv, training=training)
        maxpool = self.maxpool(bn_conv)
        squeeze = keras.backend.squeeze(maxpool, -2)
        return squeeze


    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.conv.filters})
        config.update({'kernel_size': self.conv.kernel_size})
        config.update({'padding': self.conv.padding})
        config.update({'activation': self.conv.activation})
        return config


class MusicnnMidend(keras.layers.Layer):
    def __init__(self, num_filt, **kwargs):
        super().__init__(**kwargs)
        # conv layer 1
        self.conv1 = keras.layers.Conv1D(filters=num_filt, kernel_size=7,
            padding="same", activation=keras.activations.relu, name='conv1')
        self.cnn1 = keras.layers.BatchNormalization(name='batch_norm1')

        # conv layer 2 - residual connection
        self.conv2 = keras.layers.Conv1D(filters=num_filt, kernel_size=7,
            padding="same", activation=keras.activations.relu, name='conv2')
        self.bn_conv2 = keras.layers.BatchNormalization(name='batch_norm2')
        self.cnn2 = keras.layers.Add()

        # conv layer 3 - residual connection
        self.conv3 = keras.layers.Conv1D(filters=num_filt, kernel_size=7,
            padding="same", activation=keras.activations.relu, name='conv3')
        self.bn_conv3 = keras.layers.BatchNormalization(name='batch_norm3')
        self.cnn3 = keras.layers.Add()


    def call(self, inputs, training=None):
        # conv layer 1
        conv1 = self.conv1(inputs)
        cnn1 = self.cnn1(conv1, training=training)

        # conv layer 2 - residual connection
        conv2 = self.conv2(cnn1)
        bn_conv2 = self.bn_conv2(conv2, training=training)
        cnn2 = self.cnn2([cnn1, bn_conv2])

        # conv layer 3 - residual connection
        conv3 = self.conv3(cnn2)
        bn_conv3 = self.bn_conv3(conv3, training=training)
        cnn3 = self.cnn3([cnn2, bn_conv3])

        return inputs, cnn1, cnn2, cnn3


    def get_config(self):
        config = super().get_config()
        config.update({'num_filt': self.conv1.filters})
        return config


class MusicnnBackend(keras.layers.Layer):
    def __init__(self, num_classes, output_units, dropout=0.5, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(output_units, Iterable):
            output_units = (output_units,)

        self.classifier = keras.Sequential([
            keras.layers.BatchNormalization(name='batch_norm_pool'),
            keras.layers.Dropout(rate=dropout, name='dropout_pool'),
        ])
        for idx, feature_dim in enumerate(output_units):
            self.classifier.add(keras.layers.Dense(units=feature_dim, activation=keras.activations.relu, name=f'dense{idx+1}'))
            self.classifier.add(keras.layers.BatchNormalization(name=f'batch_norm_dense{idx+1}'))
            self.classifier.add(keras.layers.Dropout(rate=dropout, name=f'dropout_dense{idx+1}'))
        # output dense layer
        self.classifier.add(keras.layers.Dense(units=num_classes, activation=None, name='logits'))


    def build(self, input_shape):
        super().build(input_shape)
        if len(input_shape) == 4:
            self.channel_last = keras.layers.Permute((2, 3, 1))
            self.max_pool = keras.layers.MaxPooling2D(pool_size=(input_shape[-2], 1), data_format='channels_last')
            self.mean_pool= keras.layers.AveragePooling2D(pool_size=(input_shape[-2], 1), data_format='channels_last')
            self.all_pool = keras.layers.Concatenate(axis=-2)
            self.flat_pool = keras.Sequential([
                keras.layers.Permute((3, 1, 2)),
                keras.layers.Lambda(lambda x: keras.backend.squeeze(x, -2)),
            ])
        else:
            self.channel_last = self.flat_pool = keras.layers.Layer()
            self.max_pool = keras.layers.GlobalMaxPooling1D(data_format='channels_last')
            self.mean_pool= keras.layers.GlobalAveragePooling1D(data_format='channels_last')
            self.all_pool = keras.layers.Concatenate(axis=-1)


    def call(self, inputs, training=None):
        # temporal pooling
        channel_last = self.channel_last(inputs)
        max_pool = self.max_pool(channel_last)
        mean_pool = self.mean_pool(channel_last)
        all_pool = self.all_pool([max_pool, mean_pool])
        flat_pool = self.flat_pool(all_pool)
        # classifier
        logits = self.classifier(flat_pool, training=training)
        return logits


    def get_config(self):
        config = super().get_config()
        config.update({'num_classes': self.classifier.get_layer('logits').units})
        config.update({'output_units': [self.classifier.get_layer(index=i).units for i in range(2, len(self.classifier.layers)-1, 3)]})
        return config


class VggBlock(keras.layers.Layer):
    def __init__(self, num_filters=32, kernel_size=(3, 3), pool_size=(2, 2), dropout_rate=0.25, strides=None, padding='same', activation=keras.activations.relu, **kwargs):
        super().__init__(**kwargs)
        self.conv = keras.layers.Conv2D(filters=num_filters, kernel_size=kernel_size,
            padding=padding, activation=activation)
        self.bn_conv = keras.layers.BatchNormalization()
        self.pool = keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides)
        self.do_pool = keras.layers.Dropout(rate=dropout_rate)


    def call(self, inputs, training=None):
        conv = self.conv(inputs)
        bn_conv = self.bn_conv(conv, training=training)
        pool = self.pool(bn_conv)
        do_pool = self.do_pool(pool, training=training)
        return pool, do_pool


    def get_config(self):
        config = super().get_config()
        config.update({'num_filters': self.conv.filters})
        config.update({'kernel_size': self.conv.kernel_size})
        config.update({'pool_size': self.pool.pool_size})
        config.update({'dropout_rate': self.do_pool.rate})
        config.update({'strides': self.conv.strides})
        config.update({'padding': self.conv.padding})
        config.update({'activation': self.conv.activation})
        return config


class Vgg(keras.Model):
    def __init__(self, num_classes, num_filters=32, **kwargs):
        super().__init__(**kwargs)
        self.bn_input = keras.layers.BatchNormalization()

        self.vgg1 = VggBlock(num_filters, pool_size=(4, 1), strides=(2, 2), name='vgg1')
        self.vgg2 = VggBlock(num_filters, name='vgg2')
        self.vgg3 = VggBlock(num_filters, name='vgg3')
        self.vgg4 = VggBlock(num_filters, name='vgg4')
        self.vgg5 = VggBlock(num_filters, pool_size=(4, 4), dropout_rate=0.5, name='vgg5')

        self.flat_vgg5 = keras.layers.Flatten()
        self.output = keras.layers.Dense(units=num_classes, activation=keras.activations.sigmoid)


    def call(self, inputs, training=None):
        bn_input = self.bn_input(inputs, training=training)

        vgg1, do_vgg1 = self.vgg1(bn_input, training=training)
        vgg2, do_vgg2 = self.vgg2(do_vgg1, training=training)
        vgg3, do_vgg3 = self.vgg3(do_vgg2, training=training)
        vgg4, do_vgg4 = self.vgg4(do_vgg3, training=training)
        vgg5, do_vgg5 = self.vgg5(do_vgg4, training=training)

        flat_vgg5 = self.flat_vgg5(do_vgg5)
        output = self.output(flat_vgg5)

        return output, vgg1, vgg2, vgg3, vgg4, vgg5   


    def get_config(self):
        config = super().get_config()
        config.update({'num_classes': self.output.units})
        config.update({'num_filters': self.vgg1.get_config()['num_filters']})
        return config
