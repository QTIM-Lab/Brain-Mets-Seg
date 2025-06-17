# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#NOTE: linear upsampling causes lots of constant folding warnings at compile time (and runs noticeably slower at run time than if using transposed convolutions) due to some issues with average pooling

import nv_norms
import tensorflow as tf
import tensorflow_addons as tfa

convolutions = {
    "Conv2d": tf.keras.layers.Conv2D,
    "Conv3d": tf.keras.layers.Conv3D,
    "ConvTranspose2d": tf.keras.layers.Conv2DTranspose,
    "ConvTranspose3d": tf.keras.layers.Conv3DTranspose,
}


class KaimingNormal(tf.keras.initializers.VarianceScaling):
    def __init__(self, negative_slope, seed=None):
        super().__init__(scale=2.0 / (1 + negative_slope**2), mode="fan_in", distribution="untruncated_normal", seed=seed)

    def get_config(self):
        return {"seed": self.seed}


def get_norm(name):
    if "group" in name:
        return tfa.layers.GroupNormalization(32, axis=-1, scale=True, center=True)
    elif "batch" in name:
        return tf.keras.layers.BatchNormalization(axis=-1, scale=True, center=True)
    elif "atex_instance" in name:
        return nv_norms.InstanceNormalization(axis=-1)
    elif "instance" in name:
        return tfa.layers.InstanceNormalization(axis=-1, scale=True, center=True)
    elif "none" in name:
        return tf.identity
    else:
        raise ValueError("Invalid normalization layer")


def extract_args(kwargs):
    args = {}
    if "input_shape" in kwargs:
        args["input_shape"] = kwargs["input_shape"]
    return args


def get_conv(filters, kernel_size, stride, dim, use_bias=True, **kwargs):
    conv = convolutions[f"Conv{dim}d"]
    return conv(
        filters=filters,
        kernel_size=kernel_size,
        strides=stride,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=KaimingNormal(kwargs["negative_slope"]),
        data_format="channels_last",
        **extract_args(kwargs),
    )


def get_transp_conv(filters, kernel_size, stride, dim, use_bias=True, **kwargs):
    conv = convolutions[f"ConvTranspose{dim}d"]
    return conv(
        filters=filters,
        kernel_size=kernel_size,
        strides=stride,
        padding="same",
        use_bias=use_bias,
        data_format="channels_last",
        **extract_args(kwargs),
    )


def get_upsample(filters, kernel_size, stride, dim, use_bias=True, **kwargs):
    conv = convolutions[f"Conv{dim}d"]
    return conv(
        filters=filters,
        kernel_size=kernel_size,
        strides=stride,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=KaimingNormal(kwargs["negative_slope"]),
        data_format="channels_last",
        **extract_args(kwargs),
    )


class ConvLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, stride, **kwargs):
        super().__init__()
        self.conv = get_conv(filters, kernel_size, stride, **kwargs)
        self.norm = get_norm(kwargs["norm"])
        self.relu = tf.keras.layers.ReLU(negative_slope=kwargs["negative_slope"])

    def call(self, data):
        out = self.conv(data)
        out = self.norm(out)
        out = self.relu(out)
        return out


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, stride, **kwargs):
        super().__init__()
        self.conv1 = ConvLayer(filters, kernel_size, stride, **kwargs)
        kwargs.pop("input_shape", None)
        self.conv2 = ConvLayer(filters, kernel_size, 1, **kwargs)

    def call(self, input_data):
        out = self.conv1(input_data)
        out = self.conv2(out)
        return out


class UpsampleBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, stride, **kwargs):
        super().__init__()
        self.transp_conv = get_transp_conv(filters, stride, stride, **kwargs)
        self.conv_block = ConvBlock(filters, kernel_size, 1, **kwargs)

    def call(self, input_data, skip_data):
        out = self.transp_conv(input_data)
        out = tf.concat((out, skip_data), axis=-1)
        out = self.conv_block(out)
        return out


class OutputBlock(tf.keras.layers.Layer):
    def __init__(self, filters, dim, negative_slope, use_bias=True):
        super().__init__()
        self.conv = get_conv(
            filters,
            kernel_size=1,
            stride=1,
            dim=dim,
            use_bias=use_bias,
            negative_slope=negative_slope,
        )

    def call(self, data):
        return self.conv(data)


#helper function to make normalization layer
def instantiate_normalization_layer(num_filters, prior_filter_num, norm_name, epsilon=0.00001, scale=True, center=True):
    if prior_filter_num is None:
        prior_filter_num = num_filters
    if "batch" in norm_name:
        return tf.keras.layers.BatchNormalization(scale=scale, center=center, epsilon=epsilon)
        #return tf.keras.layers.BatchNormalization(scale=scale, center=center, trainable=False, epsilon=epsilon)
    elif "atex_instance" in norm_name:
        return nv_norms.InstanceNormalization(axis=-1)
    elif "instance" in norm_name:
        return tfa.layers.InstanceNormalization(axis=-1, scale=scale, center=center, epsilon=epsilon)

#helper function to apply normalization layers
def apply_normalization(x, normalization_operation, training):
    #return normalization_operation(x, training=False)
    return normalization_operation(x, training=training)

class linear_upsampling(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        #
        self.upsample_1 = tf.keras.layers.UpSampling3D(size=2)
        self.upsample_2 = tf.keras.layers.AveragePooling3D(pool_size=2, strides=1, padding='same')
    #
    def call(self, x, training=True):
        x = self.upsample_1(x)
        x = self.upsample_2(x)
        return x

class conv_upsampling(tf.keras.layers.Layer):
    def __init__(self, num_filters, prior_filter_num, use_bias=False, bias_initializer='zeros', **kwargs):
        super().__init__()
        #
        kernel_initializer=KaimingNormal(kwargs["negative_slope"])
        self.norm = instantiate_normalization_layer(num_filters, prior_filter_num, kwargs["norm"])
        self.act = tf.keras.layers.ReLU(negative_slope=kwargs["negative_slope"])
        self.transp_conv = tf.keras.layers.Conv3DTranspose(num_filters, kernel_size=2, strides=2, padding="same", activation=None, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
    #
    def call(self, x, training=True):
        x = apply_normalization(x, self.norm, training)
        x = self.act(x)
        x = self.transp_conv(x)
        return x

class conv_downsampling(tf.keras.layers.Layer):
    def __init__(self, num_filters, prior_filter_num, use_bias=False, bias_initializer='zeros', **kwargs):
        super().__init__()
        #
        kernel_initializer=KaimingNormal(kwargs["negative_slope"])
        self.norm = instantiate_normalization_layer(num_filters, prior_filter_num, kwargs["norm"])
        self.act = tf.keras.layers.ReLU(negative_slope=kwargs["negative_slope"])
        self.strided_conv = tf.keras.layers.Conv3D(num_filters, kernel_size=2, strides=2, padding="same", activation=None, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
    #
    def call(self, x, training=True):
        x = apply_normalization(x, self.norm, training)
        x = self.act(x)
        x = self.strided_conv(x)
        return x

class norm_act_conv_pool_upsample(tf.keras.layers.Layer):
    def __init__(self, num_filters=32, prior_filter_num=None, kernel_size=3, use_norm_act=True, use_pooling=True, use_upsample=True, conv1=False, use_bias=False, bias_initializer='zeros', **kwargs):
        super().__init__()
        #
        self.use_norm_act = use_norm_act
        self.use_pooling = use_pooling
        self.use_upsample = use_upsample
        self.conv_kernel_size_1 = conv1
        kernel_initializer=KaimingNormal(kwargs["negative_slope"])
        #
        if self.use_norm_act == True:
            self.norm1 = instantiate_normalization_layer(num_filters, prior_filter_num, kwargs["norm"])
            self.act1 = tf.keras.layers.ReLU(negative_slope=kwargs["negative_slope"])
        self.conv1 = tf.keras.layers.Conv3D(num_filters, kernel_size=kernel_size, strides=1, padding='same', activation=None, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, **extract_args(kwargs))
        if self.use_pooling == True:
            self.pooling = tf.keras.layers.MaxPool3D(pool_size=2)
        if self.use_upsample == True:
            self.upsample_operations = conv_upsampling(num_filters, prior_filter_num, use_bias, **kwargs)
        if self.conv_kernel_size_1 == True:
            self.norm2 = instantiate_normalization_layer(num_filters, None, kwargs["norm"])
            self.act2 = tf.keras.layers.ReLU(negative_slope=kwargs["negative_slope"])
            self.conv2 = tf.keras.layers.Conv3D(num_filters // 2, kernel_size=1, strides=1, padding='same', activation=None, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
    #
    def call(self, x, training=True):
        if self.use_norm_act == True:
            x = apply_normalization(x, self.norm1, training)
            x = self.act1(x)
        x = self.conv1(x)
        if self.use_pooling == True:
            x_pool = self.pooling(x)
            return x, x_pool
        elif self.use_upsample == True:
            if self.conv_kernel_size_1 == True:
                x = apply_normalization(x, self.norm2, training)
                x = self.act2(x)
                x = self.conv2(x)
            x_upsample = self.upsample_operations(x)
            return x, x_upsample 
        else:
            return x

class conv_layer_encoder(tf.keras.layers.Layer):
    def __init__(self, num_filters=32, use_norm_act=True, use_pooling=True, **kwargs):
        super().__init__()
        #
        num_convs_per_layer = 2
        self.use_pooling = use_pooling
        self.level1 = norm_act_conv_pool_upsample(num_filters=num_filters, use_norm_act=use_norm_act, use_pooling=False, use_upsample=False, **kwargs)
        kwargs.pop("input_shape", None)
        self.level2 = norm_act_conv_pool_upsample(num_filters=num_filters, use_norm_act=True, use_pooling=self.use_pooling, use_upsample=False, **kwargs)
        #
    def call(self, x, training=True):
        x = self.level1(x)
        if self.use_pooling == True:
            skip, x_pool = self.level2(x)
            return skip, x_pool
        else:
            skip = self.level2(x)
            return skip

class encoder_block(tf.keras.layers.Layer):
    def __init__(self, num_filters=32, **kwargs):
        super().__init__()
        #
        if not isinstance(num_filters, list):
            num_filters = [num_filters]*4
        self.level1 = conv_layer_encoder(num_filters=num_filters[0], use_pooling=True, **kwargs)
        self.level2 = conv_layer_encoder(num_filters=num_filters[1], use_pooling=True, **kwargs)
        self.level3 = conv_layer_encoder(num_filters=num_filters[2], use_pooling=True, **kwargs)
        self.level4 = conv_layer_encoder(num_filters=num_filters[3], use_pooling=False, **kwargs)
    #
    def call(self, x, training=True):
        skip1, x_pool = self.level1(x)
        skip2, x_pool = self.level2(x_pool)
        skip3, x_pool = self.level3(x_pool)
        skip4 = self.level4(x_pool)
        return skip1, skip2, skip3, skip4

class conv_layer_decoder(tf.keras.layers.Layer):
    def __init__(self, current_level_num_feature, next_level_num_features, reduce_features=True, use_upsample=True, **kwargs):
        super().__init__()
        #
        num_convs_per_layer = 2
        self.reduce_features = reduce_features
        self.use_upsample = use_upsample
        self.level1 = norm_act_conv_pool_upsample(num_filters=current_level_num_feature, use_pooling=False, use_upsample=False, conv1=False, **kwargs)
        self.level2 = norm_act_conv_pool_upsample(num_filters=current_level_num_feature, use_pooling=False, use_upsample=False, conv1=False, **kwargs)
        if self.reduce_features == True:
            self.level3 = norm_act_conv_pool_upsample(num_filters=next_level_num_features, prior_filter_num=current_level_num_feature, kernel_size=1, use_pooling=False, use_upsample=self.use_upsample, conv1=False, **kwargs)
        #
    def call(self, x, training=True):
        x = self.level1(x)
        x = self.level2(x)
        if self.reduce_features == True:
            if self.use_upsample == True:
                _, x_upsample = self.level3(x)
                return x_upsample
            else:
                x = self.level3(x)
                return x
        else:
            return x

class decoder_block(tf.keras.layers.Layer):
    def __init__(self, current_level_num_feature, next_level_num_features, **kwargs):
        super().__init__()
        #
        if not isinstance(current_level_num_feature, list):
            current_level_num_feature = [current_level_num_feature]*4
        if not isinstance(next_level_num_features, list):
            next_level_num_features = [next_level_num_features]*4
        #
        #reduce features and upsample
        self.level4 = norm_act_conv_pool_upsample(num_filters=next_level_num_features[0], prior_filter_num=next_level_num_features[0], kernel_size=1, use_pooling=False, use_upsample=True, conv1=False, **kwargs)
        self.level3 = conv_layer_decoder(current_level_num_feature[1], next_level_num_features[1], reduce_features=True, use_upsample=True, **kwargs)
        self.level2 = conv_layer_decoder(current_level_num_feature[2], next_level_num_features[2], reduce_features=True, use_upsample=True, **kwargs)
        self.level1 = conv_layer_decoder(current_level_num_feature[3], next_level_num_features[3], reduce_features=True, use_upsample=False, **kwargs)
    #
    def call(self, skip4, skip3, skip2, skip1, training=True):
        _, x_upsample = self.level4(skip4)
        x_upsample = tf.concat([x_upsample, skip3], axis=-1)
        #
        x_upsample = self.level3(x_upsample)
        x_upsample = tf.concat([x_upsample, skip2], axis=-1)
        #
        x_upsample = self.level2(x_upsample)
        x_upsample = tf.concat([x_upsample, skip1], axis=-1)
        #
        x = self.level1(x_upsample)
        return x
