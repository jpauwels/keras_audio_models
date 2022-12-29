from tensorflow import keras

def build_musicnn_classifier(inputs, num_classes=2, num_units_classifier=100, final_activation=keras.activations.sigmoid, num_filt_frontend=1.6, num_filt_midend=64, num_units_backend=200, training=None, **kwargs):
    musicnn = build_musicnn(inputs, num_units_classifier, num_filt_frontend, num_filt_midend, num_units_backend, training, **kwargs)
    output = keras.layers.Dense(num_classes, activation=final_activation, name='classifier')(musicnn.get_layer('backend').output)
    return keras.Model(inputs=musicnn.input, outputs=output, name='musicnn_classifier', **kwargs)


def build_musicnn(inputs, num_classes, num_filt_frontend=1.6, num_filt_midend=64, num_units_backend=200, training=None, **kwargs):

    ### front-end ### musically motivated CNN
    frontend_features_list = MusicnnFrontend(num_filt_frontend, name='frontend')(inputs, training)
    # concatenate features coming from the front-end
    frontend_features = keras.layers.Concatenate(axis=2, name='concat_frontend')(frontend_features_list)

    ### mid-end ### dense layers
    midend_features_list = MusicnnMidend(num_filt_midend, name='midend')(frontend_features, training)
    # dense connection: concatenate features coming from different layers of the front- and mid-end
    midend_features = keras.layers.Concatenate(axis=2, name='concat_midend')(midend_features_list)

    ### back-end ### temporal pooling
    logits = MusicnnBackend(num_classes, num_units_backend, name='backend')(midend_features, training)
    taggram = keras.layers.Activation(keras.activations.sigmoid, name='taggram')(logits)

    return keras.Model(inputs=inputs, outputs=taggram, name='musicnn', **kwargs)


class Musicnn(keras.Model):
    def __init__(self, num_classes, num_filt_frontend=1.6, num_filt_midend=64, num_units_backend=200, **kwargs):
        super().__init__(**kwargs)

        ### front-end ### musically motivated CNN
        self.frontend = MusicnnFrontend(num_filt_frontend, name='frontend')
        # concatenate features coming from the front-end
        self.frontend_concat = keras.layers.Concatenate(axis=2, name='concat_frontend')

        ### mid-end ### dense layers
        self.midend = MusicnnMidend(num_filt_midend, name='midend')
        # dense connection: concatenate features coming from different layers of the front- and mid-end
        self.midend_concat = keras.layers.Concatenate(axis=2, name='concat_midend')

        ### back-end ### temporal pooling
        self.backend = MusicnnBackend(num_classes, num_units_backend, name='backend')
        self.final_activation = keras.layers.Activation(keras.activations.sigmoid, name='taggram')


    def call(self, inputs, training=None):
        frontend_features_list = self.frontend(inputs, training)
        # concatenate features coming from the front-end
        frontend_features = self.frontend_concat(frontend_features_list)

        ### mid-end ### dense layers
        midend_features_list = self.midend(frontend_features, training)
        # dense connection: concatenate features coming from different layers of the front- and mid-end
        midend_features = self.midend_concat(midend_features_list)

        ### back-end ### temporal pooling
        logits = self.backend(midend_features, training)
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
        self.timbral_padding = keras.layers.ZeroPadding2D(padding=(3, 0), name='timbral_padding')
        self.s1 = MusicnnFrontendBlock(filters=int(num_filt*32),
            kernel_size=(128, 1), padding="same", name='s1')
        self.s2 = MusicnnFrontendBlock(filters=int(num_filt*32),
            kernel_size=(64, 1), padding="same", name='s2')
        self.s3 = MusicnnFrontendBlock(filters=int(num_filt*32),
            kernel_size=(32, 1), padding="same", name='s3')


    def build(self, input_shape):
        super().build(input_shape)
        self.add_channel = keras.layers.Reshape(target_shape=input_shape[1:3]+(1,))
        self.f74 = MusicnnFrontendBlock(filters=int(self.num_filt*128),
            kernel_size=(7, int(0.4 * input_shape[2])), padding="valid", name='f74')
        self.f77 = MusicnnFrontendBlock(filters=int(self.num_filt*128),
            kernel_size=(7, int(0.7 * input_shape[2])), padding="valid", name='f77')


    def call(self, inputs, training=None):
        add_channel = self.add_channel(inputs)
        normalized_input = self.normalized_input(add_channel, training)
        input_pad_7 =  self.timbral_padding(normalized_input)
        f74 = self.f74(input_pad_7, training)
        f77 = self.f77(input_pad_7, training)
        s1 = self.s1(normalized_input, training)
        s2 = self.s2(normalized_input, training)
        s3 = self.s3(normalized_input, training)
        return [f74, f77, s1, s2, s3]

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
        self.squeeze = keras.layers.Reshape(target_shape=(-1, filters), name='squeeze')


    def build(self, input_shape):
        super().build(input_shape)
        if self.conv.padding == 'valid':
            conv_cols = input_shape[2] - self.conv.kernel_size[1] + 1
        else:
            conv_cols = input_shape[2]
        self.maxpool = keras.layers.MaxPool2D(pool_size=(1, conv_cols), name='maxpool')


    def call(self, inputs, training=None):
        conv = self.conv(inputs)
        bn_conv = self.bn_conv(conv, training)
        maxpool = self.maxpool(bn_conv)
        squeeze = self.squeeze(maxpool)
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
        cnn1 = self.cnn1(conv1, training)

        # conv layer 2 - residual connection
        conv2 = self.conv2(cnn1)
        bn_conv2 = self.bn_conv2(conv2, training)
        cnn2 = self.cnn2([cnn1, bn_conv2])

        # conv layer 3 - residual connection
        conv3 = self.conv3(cnn2)
        bn_conv3 = self.bn_conv3(conv3, training)
        cnn3 = self.cnn3([cnn2, bn_conv3])

        return inputs, cnn1, cnn2, cnn3


    def get_config(self):
        config = super().get_config()
        config.update({'num_filt': self.conv1.filters})
        return config


class MusicnnBackend(keras.layers.Layer):
    def __init__(self, num_classes, output_units, **kwargs):
        super().__init__(**kwargs)
        # temporal pooling
        self.max_pool = keras.layers.GlobalMaxPool1D()
        self.mean_pool= keras.layers.GlobalAveragePooling1D()
        self.all_pool = keras.layers.Lambda(lambda x: keras.backend.stack(x, axis=2), name='pool_concat') # keras.layers.Concatenate(axis=2)

        # penultimate dense layer
        self.flat_pool = keras.layers.Flatten()
        self.bn_flat_pool = keras.layers.BatchNormalization(name='batch_norm_pool')
        self.flat_pool_dropout = keras.layers.Dropout(rate=0.5)
        self.penultimate = keras.layers.Dense(units=output_units, activation=keras.activations.relu, name='penultimate')
        self.bn_penultimate = keras.layers.BatchNormalization(name='batch_norm_penultimate')
        self.penultimate_dropout = keras.layers.Dropout(rate=0.5)

        # output dense layer
        self.logits = keras.layers.Dense(units=num_classes, activation=None, name='logits')


    def call(self, inputs, training=None):
        # temporal pooling
        max_pool = self.max_pool(inputs)
        mean_pool = self.mean_pool(inputs)
        all_pool = self.all_pool([max_pool, mean_pool])

        # penultimate dense layer
        flat_pool = self.flat_pool(all_pool)
        bn_flat_pool = self.bn_flat_pool(flat_pool, training)
        flat_pool_dropout = self.flat_pool_dropout(bn_flat_pool, training)
        penultimate = self.penultimate(flat_pool_dropout)
        bn_penultimate = self.bn_penultimate(penultimate, training)
        penultimate_dropout = self.penultimate_dropout(bn_penultimate, training)

        # output dense layer
        logits = self.logits(penultimate_dropout)

        return logits


    def get_config(self):
        config = super().get_config()
        config.update({'num_classes': self.logits.units})
        config.update({'output_units': self.penultimate.units})
        return config


class VggBlock(keras.layers.Layer):
    def __init__(self, num_filters=32, kernel_size=(3, 3), pool_size=(2, 2), dropout_rate=0.25, strides=None, padding='same', activation=keras.activations.relu, **kwargs):
        super().__init__(**kwargs)
        self.conv = keras.layers.Conv2D(filters=num_filters, kernel_size=kernel_size,
            padding=padding, activation=activation)
        self.bn_conv = keras.layers.BatchNormalization()
        self.pool = keras.layers.MaxPool2D(pool_size=pool_size, strides=strides)
        self.do_pool = keras.layers.Dropout(rate=dropout_rate)


    def call(self, inputs, training=None):
        conv = self.conv(inputs)
        bn_conv = self.bn_conv(conv, training)
        pool = self.pool(bn_conv)
        do_pool = self.do_pool(pool, training)
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
        bn_input = self.bn_input(inputs, training)

        vgg1, do_vgg1 = self.vgg1(bn_input, training)
        vgg2, do_vgg2 = self.vgg2(do_vgg1, training)
        vgg3, do_vgg3 = self.vgg3(do_vgg2, training)
        vgg4, do_vgg4 = self.vgg4(do_vgg3, training)
        vgg5, do_vgg5 = self.vgg5(do_vgg4, training)

        flat_vgg5 = self.flat_vgg5(do_vgg5)
        output = self.output(flat_vgg5)

        return output, vgg1, vgg2, vgg3, vgg4, vgg5   


    def get_config(self):
        config = super().get_config()
        config.update({'num_classes': self.output.units})
        config.update({'num_filters': self.vgg1.get_config()['num_filters']})
        return config
